/*
 * test_abi.c — 用**纯 C**（不是 C++）验证 C ABI 可用
 *
 * 为什么单独写一个 C 测试：
 *   C++ 能调用一个"看起来像 C"的头文件，并不能证明它对 C 可用 ——
 *   常见失败是头文件里漏了 extern "C"、用了 C++ 关键字、
 *   或枚举/结构体在 C 下语法不合法。用 C 编译器编一遍是最直接的判定。
 *   本文件由 Makefile 的 `cc`（不是 c++）编译。
 *
 * 覆盖：版本探测、成功路径、各错误码、零长度边界、缓冲区不足。
 * 返回 0 表示全部通过；非 0 为失败项数。
 */

#include "ka_abi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__);   \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

#define CHECK_STATUS(expr, want, msg)                                     \
    do {                                                                  \
        ka_status _st = (expr);                                           \
        if (_st != (want)) {                                              \
            printf("  FAIL: %s  期望 %s，实得 %s  (%s:%d)\n", (msg),      \
                   ka_status_string(want), ka_status_string(_st),         \
                   __FILE__, __LINE__);                                   \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

/* 一根沿 z 轴、过原点、长 2、半径 0.1 的管 */
static const double kTube[KA_TUBE_STRIDE] = {
    0.0, 0.0, 0.0,   /* center */
    0.0, 0.0, 1.0,   /* direction */
    2.0, 0.1         /* length, delta */
};

static void test_abi_version(void) {
    printf("test_abi_version\n");
    CHECK(ka_abi_version() == KA_ABI_VERSION, "ka_abi_version 应等于 KA_ABI_VERSION");
    CHECK(ka_version_string() != NULL, "ka_version_string 不应为 NULL");
}

static void test_contains(void) {
    printf("test_contains\n");
    /* 原点在管内；(0.5, 0, 0) 距轴 0.5 > 0.1，在管外 */
    const double pts[6] = {0.0, 0.0, 0.0,   0.5, 0.0, 0.0};
    uint8_t mask[2] = {9, 9};
    CHECK_STATUS(ka_contains_points_batch(kTube, 1, pts, 2, mask), KA_OK, "基本调用");
    CHECK(mask[0] == 1, "原点应被包含");
    CHECK(mask[1] == 0, "(0.5,0,0) 不应被包含");
}

static void test_contains_edge(void) {
    printf("test_contains_edge\n");
    uint8_t mask[1] = {7};
    /* n_points == 0：合法空操作，不写缓冲区 */
    CHECK_STATUS(ka_contains_points_batch(kTube, 1, NULL, 0, mask), KA_OK, "零点数");
    CHECK(mask[0] == 7, "零点数时不应触碰缓冲区");
    /* n_tubes == 0：全部为 0（不是错误） */
    const double pt[3] = {0.0, 0.0, 0.0};
    uint8_t m2[1] = {9};
    CHECK_STATUS(ka_contains_points_batch(NULL, 0, pt, 1, m2), KA_OK, "零管数");
    CHECK(m2[0] == 0, "零管数时应为全 false");
    /* 空指针 */
    CHECK_STATUS(ka_contains_points_batch(kTube, 1, pt, 1, NULL), KA_ERR_NULL, "空输出");
    /* 负尺寸 */
    uint8_t m3[1];
    CHECK_STATUS(ka_contains_points_batch(kTube, -1, pt, 1, m3), KA_ERR_SIZE, "负管数");
}

static void test_voxel(void) {
    printf("test_voxel\n");
    const double bounds[KA_BOUNDS_LEN] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    double vol = -1.0;
    CHECK_STATUS(ka_voxel_union_volume(kTube, 1, bounds, 32, &vol), KA_OK, "基本调用");
    /* 管体积 = π r² L ≈ 3.14159 * 0.01 * 2 ≈ 0.0628；
     * 体素化会有偏差，这里只做量级断言（精确一致性由 Python 侧交叉测试管）。 */
    CHECK(vol > 0.0, "体积应为正");
    CHECK(vol < 8.0, "体积不应超过包围盒体积");
    CHECK(vol < 0.2, "细管体积应远小于包围盒");
}

static void test_voxel_errors(void) {
    printf("test_voxel_errors\n");
    const double bounds[KA_BOUNDS_LEN] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    const double bad[KA_BOUNDS_LEN] = {1.0, -1.0, -1.0, 1.0, -1.0, 1.0};
    double vol = 0.0;
    CHECK_STATUS(ka_voxel_union_volume(kTube, 1, bounds, 0, &vol), KA_ERR_GRID, "grid_n=0");
    CHECK_STATUS(ka_voxel_union_volume(kTube, 1, bounds, -3, &vol), KA_ERR_GRID, "grid_n<0");
    CHECK_STATUS(ka_voxel_union_volume(kTube, 1, bad, 8, &vol), KA_ERR_BOUNDS, "hi<lo");
    CHECK_STATUS(ka_voxel_union_volume(kTube, 1, bounds, 8, NULL), KA_ERR_NULL, "空输出");
    CHECK_STATUS(ka_voxel_union_volume(kTube, 1, NULL, 8, &vol), KA_ERR_NULL, "空 bounds");
    /* 零管：体积为 0 而非错误 */
    CHECK_STATUS(ka_voxel_union_volume(NULL, 0, bounds, 8, &vol), KA_OK, "零管数");
    CHECK(vol == 0.0, "零管数体积应为 0");
}

static void test_mc(void) {
    printf("test_mc\n");
    /* 4 个采样点，2 个命中 => 0.5 * box_volume */
    const double pts[12] = {
        0.0, 0.0, 0.0,   0.0, 0.0, 0.5,     /* 命中 */
        0.9, 0.0, 0.0,   0.0, 0.9, 0.0      /* 未命中 */
    };
    double vol = -1.0;
    CHECK_STATUS(ka_mc_union_volume(kTube, 1, pts, 4, 8.0, &vol), KA_OK, "基本调用");
    CHECK(fabs(vol - 4.0) < 1e-12, "命中率 1/2 * 8.0 应为 4.0");
    CHECK_STATUS(ka_mc_union_volume(kTube, 1, pts, 0, 8.0, &vol), KA_ERR_SIZE, "零采样");
    CHECK_STATUS(ka_mc_union_volume(kTube, 1, NULL, 4, 8.0, &vol), KA_ERR_NULL, "空采样");
}

static void test_mc_fused(void) {
    printf("test_mc_fused\n");
    /* kTube：z 轴管，真体积 ≈ π·0.1²·2 ≈ 0.0628，box=8 ⇒ 命中率 ≈ 0.00785。 */
    const double bounds[6] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    const double bad[6] = {1.0, -1.0, -1.0, 1.0, -1.0, 1.0};
    double vol = -1.0;

    CHECK_STATUS(ka_mc_union_volume_fused(kTube, 1, bounds, 200000, 42, &vol),
                 KA_OK, "基本调用");
    CHECK(vol > 0.05 && vol < 0.08, "20 万样本应贴近真体积 0.0628");

    /* 逐样本 RNG 只依赖 (seed,j) ⇒ 同 seed 逐位确定。 */
    double vol2 = -1.0;
    CHECK_STATUS(ka_mc_union_volume_fused(kTube, 1, bounds, 200000, 42, &vol2),
                 KA_OK, "同 seed 再跑");
    CHECK(vol == vol2, "同 seed 结果必须逐位相同");

    /* 错误路径与边界 */
    CHECK_STATUS(ka_mc_union_volume_fused(kTube, 1, bounds, 0, 42, &vol),
                 KA_ERR_SIZE, "零采样");
    CHECK_STATUS(ka_mc_union_volume_fused(kTube, 1, NULL, 100, 42, &vol),
                 KA_ERR_NULL, "空 bounds");
    CHECK_STATUS(ka_mc_union_volume_fused(kTube, 1, bad, 100, 42, &vol),
                 KA_ERR_BOUNDS, "hi<lo");
    CHECK_STATUS(ka_mc_union_volume_fused(kTube, 1, bounds, 100, 42, NULL),
                 KA_ERR_NULL, "空输出");
    CHECK_STATUS(ka_mc_union_volume_fused(NULL, 1, bounds, 100, 42, &vol),
                 KA_ERR_NULL, "空 tubes");
    /* 零管：体积为 0 而非错误 */
    CHECK_STATUS(ka_mc_union_volume_fused(NULL, 0, bounds, 100, 42, &vol),
                 KA_OK, "零管数");
    CHECK(vol == 0.0, "零管数体积应为 0");
}

static void test_assign(void) {
    printf("test_assign\n");
    /* 一根管，bbox 占满 [-1,1]^3 */
    const double bboxes[6] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    const double lo[3] = {-1.0, -1.0, -1.0};
    const double hi[3] = {1.0, 1.0, 1.0};
    int32_t count = 0;
    CHECK_STATUS(ka_assign_tubes_count(bboxes, 1, lo, hi, 2, 2, 2, &count),
                 KA_OK, "问长度");
    CHECK(count > 0, "应分配到至少一个单元");

    int32_t *cells = (int32_t *)malloc(sizeof(int32_t) * (size_t)count);
    int32_t *tidx = (int32_t *)malloc(sizeof(int32_t) * (size_t)count);
    CHECK_STATUS(ka_assign_tubes(bboxes, 1, lo, hi, 2, 2, 2,
                                 cells, tidx, count), KA_OK, "填充");
    CHECK(tidx[0] == 0, "管下标应为 0");

    /* 两段式调用的核心价值：容量不足时必须报 KA_ERR_BUFFER 而非越界写 */
    CHECK_STATUS(ka_assign_tubes(bboxes, 1, lo, hi, 2, 2, 2,
                                 cells, tidx, count - 1), KA_ERR_BUFFER, "容量不足");
    /* 先问长度的结果必须与填充结果一致（否则两段式调用有竞态语义问题） */
    int32_t count2 = 0;
    CHECK_STATUS(ka_assign_tubes_count(bboxes, 1, lo, hi, 2, 2, 2, &count2),
                 KA_OK, "再问长度");
    CHECK(count2 == count, "两次问长度应一致");

    CHECK_STATUS(ka_assign_tubes_count(bboxes, 1, lo, hi, 0, 2, 2, &count2),
                 KA_ERR_GRID, "nx=0");
    CHECK_STATUS(ka_assign_tubes_count(bboxes, 1, lo, hi, 2, 2, 2, NULL),
                 KA_ERR_NULL, "空 count 输出");
    free(cells);
    free(tidx);
}

static void test_status_string(void) {
    printf("test_status_string\n");
    CHECK(strcmp(ka_status_string(KA_OK), "ok") == 0, "KA_OK 文案");
    CHECK(ka_status_string((ka_status)999) != NULL, "未知码不应返回 NULL");
}

int main(void) {
    printf("== ka_abi C 语言调用测试（ABI v%d）==\n", KA_ABI_VERSION);
    test_abi_version();
    test_contains();
    test_contains_edge();
    test_voxel();
    test_voxel_errors();
    test_mc();
    test_mc_fused();
    test_assign();
    test_status_string();

    if (g_failures == 0) {
        printf("\n全部通过。\n");
        return 0;
    }
    printf("\n%d 项失败。\n", g_failures);
    return g_failures;
}
