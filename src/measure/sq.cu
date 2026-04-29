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
Calculate the total scattering function from g(r)
--------------------------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "sq.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#include <map>

namespace
{
const double SQ_PI = 3.14159265358979323846;

const std::map<std::string, double> NEUTRON_SCATTERING_LENGTH_TABLE{
  {"H", -3.7409}, {"He", 3.0985},  {"Li", -1.9300}, {"Be", 7.7900},  {"B", 5.3000},
  {"C", 6.6472},  {"N", 9.3600},   {"O", 5.8037},   {"F", 5.6540},   {"Ne", 4.5660},
  {"Na", 3.6300}, {"Mg", 5.3750},  {"Al", 3.4490},  {"Si", 4.15071}, {"P", 5.1300},
  {"S", 2.8470},  {"Cl", 9.5792},  {"Ar", 1.9090},  {"K", 3.6700},   {"Ca", 4.7000},
  {"Sc", 12.10},  {"Ti", -3.3700}, {"V", -0.4430},  {"Cr", 3.6350},  {"Mn", -3.7500},
  {"Fe", 9.4500}, {"Co", 2.4900},  {"Ni", 10.30},   {"Cu", 7.7180},  {"Zn", 5.6800},
  {"Ga", 7.2880}, {"Ge", 8.1850},  {"As", 6.5800},  {"Se", 7.9700},  {"Br", 6.7900},
  {"Kr", 7.8100}, {"Rb", 7.0800},  {"Sr", 7.0200},  {"Y", 7.7500},   {"Zr", 7.1600},
  {"Nb", 7.0540}, {"Mo", 6.7150},  {"Tc", 6.8000},  {"Ru", 7.0200},  {"Rh", 5.9000},
  {"Pd", 5.9100}, {"Ag", 5.9220},  {"Cd", 4.8300},  {"In", 4.0650},  {"Sn", 6.2239},
  {"Sb", 5.5700}, {"Te", 5.6800},  {"I", 5.2800},   {"Xe", 4.6900},  {"Cs", 5.4200},
  {"Ba", 5.0700}, {"La", 8.2400},  {"Ce", 4.8400},  {"Pr", 4.4400},  {"Nd", 7.8700},
  {"Pm", 12.60},  {"Sm", 0.0000},  {"Eu", 5.3000},  {"Gd", 9.5000},  {"Tb", 7.3400},
  {"Dy", 16.90},  {"Ho", 8.4400},  {"Er", 7.7900},  {"Tm", 7.0700},  {"Yb", 12.41},
  {"Lu", 7.2100}, {"Hf", 7.7700},  {"Ta", 6.9100},  {"W", 4.7550},   {"Re", 9.2000},
  {"Os", 10.70},  {"Ir", 10.60},   {"Pt", 9.6000},  {"Au", 7.9000},  {"Hg", 12.60},
  {"Tl", 8.7760}, {"Pb", 9.4024},  {"Bi", 8.5242},  {"Ra", 10.00},   {"Th", 10.31},
  {"Pa", 9.1000}, {"U", 8.4170},   {"Np", 10.55},   {"Pu", 7.7000},  {"Am", 8.3000},
  {"Cm", 9.5000}};

double get_neutron_scattering_length(const std::string& symbol)
{
  const auto it = NEUTRON_SCATTERING_LENGTH_TABLE.find(symbol);
  if (it == NEUTRON_SCATTERING_LENGTH_TABLE.end()) {
    std::string message =
      "No natural-abundance coherent neutron scattering length is available for element " + symbol +
      ".\n";
    PRINT_INPUT_ERROR(message.c_str());
  }
  return it->second;
}

void write_gr_header(FILE* fid, const std::vector<std::string>& type_symbols)
{
  fprintf(fid, "#radius total");
  for (int a = 0; a < type_symbols.size(); ++a) {
    for (int b = a; b < type_symbols.size(); ++b) {
      fprintf(fid, " %s-%s", type_symbols[a].c_str(), type_symbols[b].c_str());
    }
  }
  fprintf(fid, "\n");
}

void write_q_header(FILE* fid)
{
  fprintf(fid, "#Q total\n");
}

__global__ void gpu_find_rdf_ON1(
  const int N,
  const SQ::SQ_Para para,
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
              for (int w = 0; w < para.num_bins; ++w) {
                const double r_low = (w * para.dr) * (w * para.dr);
                const double r_up = ((w + 1) * para.dr) * ((w + 1) * para.dr);
                const double r_mid_square = ((w + 0.5) * para.dr) * ((w + 0.5) * para.dr);
                const double dV = r_mid_square * 4.0 * 3.14159265358979323846 * para.dr;
                if (d2 > r_low && d2 <= r_up) {
                  atomicAdd(&rdf_[w * para.num_SQs + 0], 1 / (N * para.density_global * dV));
                  int count = 1;
                  for (int a = 0; a < para.num_types; ++a) {
                    for (int b = a; b < para.num_types; ++b) {
                      if (type[n1] == para.type_index[a] && type[n2] == para.type_index[b]) {
                        atomicAdd(
                          &rdf_[w * para.num_SQs + count],
                          1 / (para.num_atoms[a] * para.density_type[b] * dV));
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

void SQ::find_rdf(Box& box, const GPU_Vector<int>& type, const GPU_Vector<double>& position)
{
  const int N = type.size();
  const double rc_cell_list = 0.5 * sq_para.rc;
  const double rc_inv_cell_list = 2.0 / sq_para.rc;
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
    sq_para,
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
    rdf_g_.data());
  GPU_CHECK_KERNEL
}

void SQ::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  rdf_g_.resize(sq_para.num_SQs * sq_para.num_bins, 0);
  cell_count.resize(atom.number_of_atoms);
  cell_count_sum.resize(atom.number_of_atoms);
  cell_contents.resize(atom.number_of_atoms);

  type_symbols_.resize(sq_para.num_types);
  for (int a = 0; a < sq_para.num_types; ++a) {
    const int type_idx = sq_para.type_index[a];
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      if (atom.cpu_type[n] == type_idx) {
        type_symbols_[a] = atom.cpu_atom_symbol[n];
        break;
      }
    }
  }

  neutron_scattering_length_.assign(sq_para.num_types, 0.0);
  average_neutron_scattering_length_ = 0.0;
  average_neutron_scattering_length_square_ = 0.0;
  average_neutron_scattering_length_squared_mean_ = 0.0;

  if (weight_mode_ == 0) {
    printf("    weight = 0: output unweighted total scattering.\n");
    return;
  }

  printf("    weight = 1: output neutron-weighted total scattering.\n");
  for (int a = 0; a < sq_para.num_types; ++a) {
    neutron_scattering_length_[a] = get_neutron_scattering_length(type_symbols_[a]);
    const double concentration = static_cast<double>(sq_para.num_atoms[a]) / atom.number_of_atoms;
    average_neutron_scattering_length_ += concentration * neutron_scattering_length_[a];
    average_neutron_scattering_length_squared_mean_ +=
      concentration * neutron_scattering_length_[a] * neutron_scattering_length_[a];
    printf(
      "        Type %d (%s) has neutron coherent scattering length %g fm.\n",
      sq_para.type_index[a],
      type_symbols_[a].c_str(),
      neutron_scattering_length_[a]);
  }

  average_neutron_scattering_length_square_ =
    average_neutron_scattering_length_ * average_neutron_scattering_length_;
  if (fabs(average_neutron_scattering_length_square_) < 1.0e-12) {
    PRINT_INPUT_ERROR("Average neutron coherent scattering length is too close to zero.\n");
  }
}

void SQ::process(
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

  sq_para.volume = box.get_volume();
  sq_para.density_global = atom.number_of_atoms / sq_para.volume;
  for (int t = 0; t < sq_para.num_types; ++t) {
    sq_para.density_type[t] = sq_para.num_atoms[t] / sq_para.volume;
  }
  find_rdf(box, atom.type, integrate.type >= 31 ? atom.position_beads[0] : atom.position_per_atom);
}

void SQ::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  std::vector<double> rdf_(sq_para.num_SQs * sq_para.num_bins, 0.0);
  std::vector<double> total_gr(sq_para.num_bins, 0.0);
  rdf_g_.copy_to_host(rdf_.data());

  FILE* fid_gr = my_fopen("SQ_gr.out", "a");
  FILE* fid_sq = my_fopen("SQ_SQ.out", "a");
  FILE* fid_fq = my_fopen("SQ_FQ.out", "a");
  FILE* fid_iq = my_fopen("SQ_IQ.out", "a");

  write_gr_header(fid_gr, type_symbols_);
  write_q_header(fid_sq);
  write_q_header(fid_fq);
  write_q_header(fid_iq);

  const int num_repeats = number_of_steps / sampling_interval_;
  for (int bin = 0; bin < sq_para.num_bins; ++bin) {
    const double r = bin * sq_para.dr + sq_para.dr / 2.0;
    const double g_total_unweighted = rdf_[bin * sq_para.num_SQs + 0] / num_repeats;
    double g_total = g_total_unweighted;
    if (weight_mode_ == 1) {
      g_total = 0.0;
      int count = 1;
      for (int a = 0; a < sq_para.num_types; ++a) {
        const double concentration_a = static_cast<double>(sq_para.num_atoms[a]) / atom.number_of_atoms;
        for (int b = a; b < sq_para.num_types; ++b) {
          const double concentration_b =
            static_cast<double>(sq_para.num_atoms[b]) / atom.number_of_atoms;
          const double pair_multiplicity = (a == b) ? 1.0 : 2.0;
          const double pair_weight =
            pair_multiplicity * concentration_a * concentration_b *
            neutron_scattering_length_[a] * neutron_scattering_length_[b] /
            average_neutron_scattering_length_square_;
          const double g_ab = rdf_[bin * sq_para.num_SQs + count] / num_repeats;
          g_total += pair_weight * g_ab;
          ++count;
        }
      }
    }
    total_gr[bin] = g_total;

    fprintf(fid_gr, "%.5f %.5f", r, g_total);
    int count = 1;
    for (int a = 0; a < sq_para.num_types; ++a) {
      for (int b = a; b < sq_para.num_types; ++b) {
        const double g_ab = rdf_[bin * sq_para.num_SQs + count] / num_repeats;
        fprintf(fid_gr, " %.5f", g_ab);
        ++count;
      }
    }
    fprintf(fid_gr, "\n");
  }

  const double density = sq_para.density_global;
  for (int q_bin = 0; q_bin < q_num_bins_; ++q_bin) {
    const double q = q_bin * dq_ + dq_ / 2.0;
    double sq_total = 1.0;
    for (int r_bin = 0; r_bin < sq_para.num_bins; ++r_bin) {
      const double r = r_bin * sq_para.dr + sq_para.dr / 2.0;
      const double qr = q * r;
      const double sinc = (fabs(qr) < 1.0e-12) ? 1.0 : sin(qr) / qr;
      sq_total += 4.0 * SQ_PI * density * r * r * (total_gr[r_bin] - 1.0) * sinc * sq_para.dr;
    }
    const double fq_total = q * (sq_total - 1.0);
    const double iq_total =
      (weight_mode_ == 1)
        ? average_neutron_scattering_length_squared_mean_ +
            average_neutron_scattering_length_square_ * (sq_total - 1.0)
        : sq_total;
    fprintf(fid_sq, "%.5f %.5f\n", q, sq_total);
    fprintf(fid_fq, "%.5f %.5f\n", q, fq_total);
    fprintf(fid_iq, "%.5f %.5f\n", q, iq_total);
  }

  fflush(fid_gr);
  fflush(fid_sq);
  fflush(fid_fq);
  fflush(fid_iq);
  fclose(fid_gr);
  fclose(fid_sq);
  fclose(fid_fq);
  fclose(fid_iq);
}

SQ::SQ(
  const char** param,
  const int num_param,
  Box& box,
  const std::vector<int>& cpu_type_size,
  const int number_of_steps)
{
  parse(param, num_param, box, cpu_type_size, number_of_steps);
  property_name = "compute_sq";
}

void SQ::parse(
  const char** param,
  const int num_param,
  Box& box,
  const std::vector<int>& cpu_type_size,
  const int number_of_steps)
{
  printf("Compute total scattering function (SQ).\n");

  if (num_param != 7) {
    PRINT_INPUT_ERROR("compute_sq should have 6 parameters.\n");
  }

  if (!is_valid_real(param[1], &sq_para.rc)) {
    PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
  }
  if (sq_para.rc <= 0) {
    PRINT_INPUT_ERROR("radial cutoff should be positive.\n");
  }
  double thickness_half[3] = {
    box.get_volume() / box.get_area(0) / 2.5,
    box.get_volume() / box.get_area(1) / 2.5,
    box.get_volume() / box.get_area(2) / 2.5};
  if (sq_para.rc > thickness_half[0] || sq_para.rc > thickness_half[1] ||
      sq_para.rc > thickness_half[2]) {
    std::string message =
      "The box has a thickness < 2.5 SQ radial cutoffs in a periodic direction.\n"
      "                Please increase the periodic direction(s).\n";
    PRINT_INPUT_ERROR(message.c_str());
  }
  printf("    radial cutoff %g.\n", sq_para.rc);

  if (!is_valid_int(param[2], &sq_para.num_bins)) {
    PRINT_INPUT_ERROR("number of bins should be an integer.\n");
  }
  if (sq_para.num_bins <= 20) {
    PRINT_INPUT_ERROR("A larger nbins is recommended.\n");
  }
  if (sq_para.num_bins > 500) {
    PRINT_INPUT_ERROR("A smaller nbins is recommended.\n");
  }
  printf("    radial cutoff will be divided into %d bins.\n", sq_para.num_bins);

  if (!is_valid_int(param[3], &sampling_interval_)) {
    PRINT_INPUT_ERROR("interval step per sample should be an integer.\n");
  }
  if (sampling_interval_ <= 0) {
    PRINT_INPUT_ERROR("interval step per sample should be positive.\n");
  }
  printf("    SQ sample interval is %d step.\n", sampling_interval_);

  if (!is_valid_real(param[4], &q_cutoff_)) {
    PRINT_INPUT_ERROR("Q cutoff should be a number.\n");
  }
  if (q_cutoff_ <= 0.0) {
    PRINT_INPUT_ERROR("Q cutoff should be positive.\n");
  }
  printf("    Q cutoff %g 1/A.\n", q_cutoff_);

  if (!is_valid_int(param[5], &q_num_bins_)) {
    PRINT_INPUT_ERROR("number of Q bins should be an integer.\n");
  }
  if (q_num_bins_ <= 0) {
    PRINT_INPUT_ERROR("number of Q bins should be positive.\n");
  }
  printf("    Q cutoff will be divided into %d bins.\n", q_num_bins_);

  if (!is_valid_int(param[6], &weight_mode_)) {
    PRINT_INPUT_ERROR("weight should be an integer.\n");
  }
  if (weight_mode_ != 0 && weight_mode_ != 1) {
    PRINT_INPUT_ERROR("weight should be 0 or 1.\n");
  }

  sq_para.num_types = 0;
  for (int t = 0; t < cpu_type_size.size(); ++t) {
    if (cpu_type_size[t] != 0) {
      sq_para.type_index[sq_para.num_types] = t;
      sq_para.num_atoms[sq_para.num_types] = cpu_type_size[t];
      sq_para.num_types++;
    }
  }
  sq_para.num_SQs = 1 + (sq_para.num_types * (sq_para.num_types + 1)) / 2;
  sq_para.rc_square = sq_para.rc * sq_para.rc;
  sq_para.dr = sq_para.rc / sq_para.num_bins;
  dq_ = q_cutoff_ / q_num_bins_;

  printf("    There are %d atom types in model.xyz.\n", sq_para.num_types);
  for (int a = 0; a < sq_para.num_types; ++a) {
    printf("        Type %d has %d atoms.\n", sq_para.type_index[a], sq_para.num_atoms[a]);
  }
  printf("    Will calculate one total g(r) and %d partial g(r)s.\n", sq_para.num_SQs - 1);
}
