/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Calculate the reduced pair distribution function (PDF)
Initial implementation: Yong Wang
Refactored by: Zheyong Fan
--------------------------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "pdf.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

namespace
{
__global__ void gpu_find_rdf_ON1(
  const int N,
  const PDF::PDF_Para para,
  const Box box,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int* __restrict__ type,
  double* rdf_)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  double rdf_PI = 3.14159265358979323846;
  if (n1 < N) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n1 != n2) {
              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
              if (d2 > para.rc_square) {
                continue;
              }
              for (int w = 0; w < para.num_bins; w++) {
                double r_low = (w * para.dr) * (w * para.dr);
                double r_up = ((w + 1) * para.dr) * ((w + 1) * para.dr);
                double r_mid_sqaure = ((w + 0.5) * para.dr) * ((w + 0.5) * para.dr);
                double dV = r_mid_sqaure * 4 * rdf_PI * para.dr;
                if (d2 > r_low && d2 <= r_up) {
                  atomicAdd(&rdf_[w * para.num_PDFs + 0], 1 / (N * para.density_global * dV));
                  int count = 1;
                  for (int a = 0; a < para.num_types; ++a) {
                    for (int b = a; b < para.num_types; ++b) {
                      if(type[n1] == para.type_index[a] && type[n2] == para.type_index[b]) {
                        atomicAdd(&rdf_[w * para.num_PDFs + count], 1 / (para.num_atoms[a] * para.density_type[b] * dV));
                      }
                      ++count;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
} // namespace

void PDF::find_rdf(Box& box, const GPU_Vector<int>& type, const GPU_Vector<double>& position)
{
  const int N = type.size();
  const double rc_cell_list = 0.5 * pdf_para.rc;
  const double rc_inv_cell_list = 2.0 / pdf_para.rc;
  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);
  find_cell_list(
    rc_cell_list,
    num_bins,
    box,
    position,
    cell_count,
    cell_count_sum,
    cell_contents);

  gpu_find_rdf_ON1<<<(N - 1) / 256 + 1, 256>>>(
    N,
    pdf_para,
    box,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    type.data(),
    pdf_g_.data());
  GPU_CHECK_KERNEL
}

void PDF::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  pdf_g_.resize(pdf_para.num_PDFs * pdf_para.num_bins, 0);
  cell_count.resize(atom.number_of_atoms);
  cell_count_sum.resize(atom.number_of_atoms);
  cell_contents.resize(atom.number_of_atoms);
}

void PDF::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % sampling_interval_ != 0) {
    return;
  }

  pdf_para.volume = box.get_volume();
  pdf_para.density_global = atom.number_of_atoms / pdf_para.volume;
  for (int t = 0; t < pdf_para.num_types; ++ t) {
    pdf_para.density_type[t] = pdf_para.num_atoms[t] / pdf_para.volume;
  }
  find_rdf(box, atom.type, integrate.type >= 31 ? atom.position_beads[0] : atom.position_per_atom);
}

void PDF::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  std::vector<double> rdf_(pdf_para.num_PDFs * pdf_para.num_bins, 0);
  pdf_g_.copy_to_host(rdf_.data());

  FILE* fid = fopen("pdf.out", "a");

  std::vector<std::string> type_symbols(pdf_para.num_types);
  for (int a = 0; a < pdf_para.num_types; a++) {
    int type_idx = pdf_para.type_index[a];
    for (int n = 0; n < atom.number_of_atoms; n++) {
      if (atom.cpu_type[n] == type_idx) {
        type_symbols[a] = atom.cpu_atom_symbol[n];
        break;
      }
    }
  }

  fprintf(fid, "#radius total");
  for (int a = 0; a < pdf_para.num_types; a++) {
    for (int b = a; b < pdf_para.num_types; b++) {
      fprintf(fid, " %s-%s", type_symbols[a].c_str(), type_symbols[b].c_str());
    }
  }
  fprintf(fid, "\n");

  const int num_repeats = number_of_steps / sampling_interval_;
  const double pi = 3.14159265358979323846;
  for (int bin = 0; bin < pdf_para.num_bins; bin++) {
    const double r = bin * pdf_para.dr + pdf_para.dr / 2;
    const double g_total = rdf_[bin * pdf_para.num_PDFs + 0] / num_repeats;
    const double G_total = 4.0 * pi * pdf_para.density_global * r * (g_total - 1.0);
    fprintf(fid, "%.5f", r);
    fprintf(fid, " %.5f", G_total);
    int count = 1;
    for (int a = 0; a < pdf_para.num_types; a++) {
      for (int b = a; b < pdf_para.num_types; b++) {
        const double g_ab = rdf_[bin * pdf_para.num_PDFs + count] / num_repeats;
        const double G_ab = 4.0 * pi * pdf_para.density_type[b] * r * (g_ab - 1.0);
        fprintf(fid, " %.5f", G_ab);
        ++count;
      }
    }
    fprintf(fid, "\n");
  }

  fflush(fid);
  fclose(fid);
}

PDF::PDF(
  const char** param,
  const int num_param,
  Box& box,
  const std::vector<int>& cpu_type_size,
  const int number_of_steps)
{
  parse(param, num_param, box, cpu_type_size, number_of_steps);
  property_name = "compute_pdf";
}

void PDF::parse(
  const char** param,
  const int num_param,
  Box& box,
  const std::vector<int>& cpu_type_size,
  const int number_of_steps)
{
  printf("Compute reduced pair distribution function (PDF).\n");

  if (num_param != 4) {
    PRINT_INPUT_ERROR("compute_pdf should have 3 parameters.\n");
  }

  if (!is_valid_real(param[1], &pdf_para.rc)) {
    PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
  }
  if (pdf_para.rc <= 0) {
    PRINT_INPUT_ERROR("radial cutoff should be positive.\n");
  }
  double thickness_half[3] = {
    box.get_volume() / box.get_area(0) / 2.5,
    box.get_volume() / box.get_area(1) / 2.5,
    box.get_volume() / box.get_area(2) / 2.5};
  if (pdf_para.rc > thickness_half[0] || pdf_para.rc > thickness_half[1] || pdf_para.rc > thickness_half[2]) {
    std::string message =
      "The box has a thickness < 2.5 PDF radial cutoffs in a periodic direction.\n"
      "                Please increase the periodic direction(s).\n";
    PRINT_INPUT_ERROR(message.c_str());
  }
  printf("    radial cutoff %g.\n", pdf_para.rc);

  if (!is_valid_int(param[2], &pdf_para.num_bins)) {
    PRINT_INPUT_ERROR("number of bins should be an integer.\n");
  }
  if (pdf_para.num_bins <= 20) {
    PRINT_INPUT_ERROR("A larger nbins is recommended.\n");
  }

  if (pdf_para.num_bins > 500) {
    PRINT_INPUT_ERROR("A smaller nbins is recommended.\n");
  }

  printf("    radial cutoff will be divided into %d bins.\n", pdf_para.num_bins);

  if (!is_valid_int(param[3], &sampling_interval_)) {
    PRINT_INPUT_ERROR("interval step per sample should be an integer.\n");
  }
  if (sampling_interval_ <= 0) {
    PRINT_INPUT_ERROR("interval step per sample should be positive.\n");
  }
  printf("    PDF sample interval is %d step.\n", sampling_interval_);

  pdf_para.num_types = 0;
  for (int t = 0; t < cpu_type_size.size(); ++t) {
    if (cpu_type_size[t] != 0) {
      pdf_para.type_index[pdf_para.num_types] = t;
      pdf_para.num_atoms[pdf_para.num_types] = cpu_type_size[t];
      pdf_para.num_types++;
    }
  }
  pdf_para.num_PDFs = 1 + (pdf_para.num_types * (pdf_para.num_types + 1)) / 2;
  pdf_para.rc_square = pdf_para.rc * pdf_para.rc;
  pdf_para.dr = pdf_para.rc / pdf_para.num_bins;

  printf("    There are %d atom types in model.xyz.\n", pdf_para.num_types);
  for (int a = 0; a < pdf_para.num_types; ++a) {
    printf("        Type %d has %d atoms.\n", pdf_para.type_index[a], pdf_para.num_atoms[a]);
  }
  printf("    Will calculate one total PDF and %d partial PDFs.\n", pdf_para.num_PDFs - 1);
}

