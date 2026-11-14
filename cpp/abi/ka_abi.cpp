/*
 * ka_abi.cpp — C ABI 稳定接口的实现
 *
 * 契约见 ka_abi.h（改行为前先读那份文件顶部的"设计约定"）。
 *
 * 本文件是**薄适配层**：把 C 调用方传进来的裸指针 + 长度校验后，转调
 * cpp/kernel/geo_core.hpp 里的几何核心。几何算法**不再在这里重述**——
 * 此前它与 tube_kernel.cpp 各写一遍，实测已出现 `const` 等细节漂移
 * （2026-09-27）。现在两边共同包含 geo_core.hpp，唯一实现。
 */

#include "ka_abi.h"
#include "../kernel/geo_core.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

/* ---------------------------------------------------------------------------
 * 版本
 * ------------------------------------------------------------------------- */

extern "C" {

int32_t ka_abi_version(void) { return KA_ABI_VERSION; }

const char *ka_version_string(void) { return "kakeya-ai c-abi 0.1.0"; }

const char *ka_status_string(ka_status status) {
    switch (status) {
        case KA_OK:         return "ok";
        case KA_ERR_NULL:   return "null pointer argument";
        case KA_ERR_SIZE:   return "invalid size argument";
        case KA_ERR_GRID:   return "invalid grid argument";
        case KA_ERR_BOUNDS: return "invalid bounds (hi <= lo)";
        case KA_ERR_ABI:    return "ABI version mismatch";
        case KA_ERR_BUFFER: return "output buffer too small";
        default:            return "unknown status";
    }
}

} /* extern "C" */

/* ---------------------------------------------------------------------------
 * 内部：几何核心的别名（实现见 geo_core.hpp）
 * ------------------------------------------------------------------------- */

namespace {

using ka_geo::build_tube_data;
using ka_geo::point_in_any;
using ka_geo::point_segment_dist2;
using ka_geo::TubeData;
using ka_geo::TUBE_STRIDE;
using ka_geo::voxel_union_volume;

} /* namespace */

/* ---------------------------------------------------------------------------
 * 1. 批量管-点包含测试
 * ------------------------------------------------------------------------- */

extern "C" KA_API ka_status ka_contains_points_batch(
    const double *tubes, int32_t n_tubes,
    const double *points, int32_t n_points,
    uint8_t *out_mask)
{
    if (out_mask == nullptr) return KA_ERR_NULL;
    if (n_points < 0 || n_tubes < 0) return KA_ERR_SIZE;
    /* n_points == 0 是合法的空操作，但此时 points 可为 NULL。 */
    if (n_points > 0 && points == nullptr) return KA_ERR_NULL;
    if (n_tubes > 0 && tubes == nullptr) return KA_ERR_NULL;

    /* 先全置 false：语义是"至少被一根管包含"，无管时即为全 false。 */
    for (int32_t j = 0; j < n_points; ++j) out_mask[j] = 0;
    if (n_tubes == 0 || n_points == 0) return KA_OK;

    const std::vector<TubeData> tds = build_tube_data(tubes, n_tubes);
    double t = 0.0;
    for (int32_t j = 0; j < n_points; ++j) {
        const double *p = points + static_cast<size_t>(j) * 3;
        for (const auto &td : tds) {
            const double dist2 = point_segment_dist2(
                p[0], p[1], p[2], td.ax, td.ay, td.az, td.bx, td.by, td.bz, t);
            if (dist2 <= td.delta_sq) { out_mask[j] = 1; break; }
        }
    }
    return KA_OK;
}

/* ---------------------------------------------------------------------------
 * 2. 体素网格并集体积估计
 * ------------------------------------------------------------------------- */

extern "C" KA_API ka_status ka_voxel_union_volume(
    const double *tubes, int32_t n_tubes,
    const double *bounds, int32_t grid_n,
    double *out_volume)
{
    if (out_volume == nullptr) return KA_ERR_NULL;
    if (bounds == nullptr) return KA_ERR_NULL;
    if (n_tubes < 0) return KA_ERR_SIZE;
    if (grid_n <= 0) return KA_ERR_GRID;
    if (n_tubes > 0 && tubes == nullptr) return KA_ERR_NULL;

    const double lo[3] = {bounds[0], bounds[2], bounds[4]};
    const double hi[3] = {bounds[1], bounds[3], bounds[5]};
    for (int a = 0; a < 3; ++a) {
        if (!(hi[a] > lo[a])) return KA_ERR_BOUNDS;  /* 也拦 NaN */
    }

    if (n_tubes == 0) { *out_volume = 0.0; return KA_OK; }

    /* 网格与求和都交给 geo_core.hpp —— 与 pybind11 路径共用同一份实现。 */
    const std::vector<TubeData> tds = build_tube_data(tubes, n_tubes);
    *out_volume = voxel_union_volume(tds, bounds, grid_n);
    return KA_OK;
}

/* ---------------------------------------------------------------------------
 * 3. 蒙特卡洛并集体积估计
 * ------------------------------------------------------------------------- */

extern "C" KA_API ka_status ka_mc_union_volume(
    const double *tubes, int32_t n_tubes,
    const double *points, int32_t n_samples,
    double box_volume,
    double *out_volume)
{
    if (out_volume == nullptr) return KA_ERR_NULL;
    if (points == nullptr) return KA_ERR_NULL;
    if (n_samples <= 0 || n_tubes < 0) return KA_ERR_SIZE;
    if (n_tubes > 0 && tubes == nullptr) return KA_ERR_NULL;

    if (n_tubes == 0) { *out_volume = 0.0; return KA_OK; }

    const std::vector<TubeData> tds = build_tube_data(tubes, n_tubes);

    int64_t count_inside = 0;
    for (int32_t j = 0; j < n_samples; ++j) {
        const double *p = points + static_cast<size_t>(j) * 3;
        if (point_in_any(tds, p[0], p[1], p[2])) ++count_inside;
    }
    *out_volume = static_cast<double>(count_inside) / n_samples * box_volume;
    return KA_OK;
}

extern "C" KA_API ka_status ka_mc_union_volume_fused(
    const double *tubes, int32_t n_tubes,
    const double *bounds, int64_t n_samples, uint64_t seed,
    double *out_volume)
{
    if (out_volume == nullptr) return KA_ERR_NULL;
    if (bounds == nullptr) return KA_ERR_NULL;
    if (n_samples <= 0 || n_tubes < 0) return KA_ERR_SIZE;
    if (n_tubes > 0 && tubes == nullptr) return KA_ERR_NULL;

    const double hi[3] = {bounds[1], bounds[3], bounds[5]};
    const double lo[3] = {bounds[0], bounds[2], bounds[4]};
    for (int a = 0; a < 3; ++a) {
        if (!(hi[a] > lo[a])) return KA_ERR_BOUNDS;  /* 也拦 NaN */
    }

    if (n_tubes == 0) { *out_volume = 0.0; return KA_OK; }

    const std::vector<TubeData> tds = build_tube_data(tubes, n_tubes);
    *out_volume = ka_geo::mc_union_volume_fused(tds, bounds, n_samples, seed);
    return KA_OK;
}

/* ---------------------------------------------------------------------------
 * 4. 网格单元 → 管 的分配
 * ------------------------------------------------------------------------- */

namespace {

/*
 * 核心分配逻辑，与 tube_kernel.cpp::assign_tubes_to_grid_cells 逐行对应。
 * collect == false 时只数个数（供 ka_assign_tubes_count 用），
 * 避免为了"问长度"而先分配一个可能很大的数组。
 */
ka_status assign_tubes_impl(
    const double *bboxes, int32_t n_tubes,
    const double *lo, const double *hi,
    int32_t nx, int32_t ny, int32_t nz,
    int32_t *out_cell_ids, int32_t *out_tube_idx, int32_t capacity,
    int32_t *out_count)
{
    if (out_count == nullptr) return KA_ERR_NULL;
    if (bboxes == nullptr || lo == nullptr || hi == nullptr) return KA_ERR_NULL;
    if (n_tubes < 0) return KA_ERR_SIZE;
    if (nx <= 0 || ny <= 0 || nz <= 0) return KA_ERR_GRID;
    for (int a = 0; a < 3; ++a) {
        if (!(hi[a] > lo[a])) return KA_ERR_BOUNDS;
    }

    const double lox = lo[0], loy = lo[1], loz = lo[2];
    const double stepx = (hi[0] - lox) / std::max(nx, 1);
    const double stepy = (hi[1] - loy) / std::max(ny, 1);
    const double stepz = (hi[2] - loz) / std::max(nz, 1);
    const double inv_stepx = 1.0 / stepx;
    const double inv_stepy = 1.0 / stepy;
    const double inv_stepz = 1.0 / stepz;

    int32_t written = 0;
    for (int32_t t = 0; t < n_tubes; ++t) {
        const double *bb = bboxes + static_cast<size_t>(t) * 6;
        const double xlo = bb[0], xhi = bb[1];
        const double ylo = bb[2], yhi = bb[3];
        const double zlo = bb[4], zhi = bb[5];

        const int32_t ix_min = std::max<int32_t>(0, static_cast<int32_t>(std::floor((xlo - lox) * inv_stepx)));
        const int32_t ix_max = std::min<int32_t>(nx - 1, static_cast<int32_t>(std::ceil((xhi - lox) * inv_stepx)));
        const int32_t iy_min = std::max<int32_t>(0, static_cast<int32_t>(std::floor((ylo - loy) * inv_stepy)));
        const int32_t iy_max = std::min<int32_t>(ny - 1, static_cast<int32_t>(std::ceil((yhi - loy) * inv_stepy)));
        const int32_t iz_min = std::max<int32_t>(0, static_cast<int32_t>(std::floor((zlo - loz) * inv_stepz)));
        const int32_t iz_max = std::min<int32_t>(nz - 1, static_cast<int32_t>(std::ceil((zhi - loz) * inv_stepz)));

        if (ix_min > ix_max || iy_min > iy_max || iz_min > iz_max) continue;

        for (int32_t ix = ix_min; ix <= ix_max; ++ix) {
            const double cxlo = lox + stepx * ix;
            const double cxhi = cxlo + stepx;
            if (xhi < cxlo || xlo > cxhi) continue;
            const int32_t base_x = ix * ny * nz;
            for (int32_t iy = iy_min; iy <= iy_max; ++iy) {
                const double cylo = loy + stepy * iy;
                const double cyhi = cylo + stepy;
                if (yhi < cylo || ylo > cyhi) continue;
                const int32_t base_xy = base_x + iy * nz;
                for (int32_t iz = iz_min; iz <= iz_max; ++iz) {
                    const double czlo = loz + stepz * iz;
                    const double czhi = czlo + stepz;
                    if (zhi >= czlo && zlo <= czhi) {
                        if (out_cell_ids != nullptr) {
                            if (written >= capacity) {
                                *out_count = written;
                                return KA_ERR_BUFFER;
                            }
                            out_cell_ids[written] = base_xy + iz;
                            out_tube_idx[written] = t;
                        }
                        ++written;
                    }
                }
            }
        }
    }
    *out_count = written;
    return KA_OK;
}

} /* namespace */

extern "C" KA_API ka_status ka_assign_tubes_count(
    const double *bboxes, int32_t n_tubes,
    const double *lo, const double *hi,
    int32_t nx, int32_t ny, int32_t nz,
    int32_t *out_count)
{
    return assign_tubes_impl(bboxes, n_tubes, lo, hi, nx, ny, nz,
                             nullptr, nullptr, 0, out_count);
}

extern "C" KA_API ka_status ka_assign_tubes(
    const double *bboxes, int32_t n_tubes,
    const double *lo, const double *hi,
    int32_t nx, int32_t ny, int32_t nz,
    int32_t *out_cell_ids, int32_t *out_tube_idx, int32_t capacity)
{
    if (out_cell_ids == nullptr || out_tube_idx == nullptr) return KA_ERR_NULL;
    if (!(capacity > 0)) return KA_ERR_SIZE;
    int32_t count = 0;
    return assign_tubes_impl(bboxes, n_tubes, lo, hi, nx, ny, nz,
                             out_cell_ids, out_tube_idx, capacity, &count);
}
