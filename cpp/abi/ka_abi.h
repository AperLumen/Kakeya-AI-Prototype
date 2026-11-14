/*
 * ka_abi.h — Kakeya AI 几何内核的 **C ABI 稳定接口**
 *
 * 目的
 * ----
 * 让本项目作为"基础设施底层模块"被**任何语言**调用，而不仅限于 Python。
 * 本头文件是**唯一的对外契约**：纯 C（不含 C++ 语法），不依赖 Python、
 * 不依赖 pybind11、不依赖任何第三方库。Rust / Go / Java / Julia / C# / C
 * 都能通过 FFI 直接调用。
 *
 * 与 Python 路径的关系
 * --------------------
 * 同一份几何逻辑（cpp/kernel/tube_kernel.cpp）被两条路复用：
 *
 *   Python  →  pybind11  →  内核        （pip install / build_nano.sh）
 *   任意语言 →  C ABI      →  内核        （本项目，make）
 *
 * 两条路对**同一输入必须给出逐位相同的结果**（本项目的"跨后端逐位同"
 * 不变量）。tests/ 里有交叉一致性测试钉住这一点。
 *
 * 设计约定（为什么这么定，改之前请先读）
 * --------------------------------------
 * 1. **内存所有权：调用方分配，库只写不 alloc/free。**
 *    跨语言/跨 CRT 的 malloc/free 不同源是 C ABI 最常见的崩溃源
 *    （Windows DLL 尤甚）。所有输出一律走调用方给的缓冲区，
 *    全部数据函数返回 ka_status 而**不返回裸指针**。
 * 2. **错误用状态码，不用 errno / 异常。**
 *    异常无法跨 C ABI；errno 是线程局部且语义偏窄。
 * 3. **数据布局与 Python 侧 tubes_to_array 逐字段一致**（见 ka_tube 注释），
 *    避免出现"两套语义"。
 * 4. **版本可探测。** C ABI 一旦发布不能改签名，用 ka_abi_version()
 *    把"旧调用方撞上新库"变成可检测的错误，而不是随机内存错乱。
 * 5. **只导出 ka_ 前缀的符号。** 见 KA_API 注释；编译时用
 *    -fvisibility=hidden，避免在被宿主进程（如 Rust/Go 程序）加载时污染符号表。
 */

#ifndef KA_ABI_H
#define KA_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * 版本与符号导出
 * ------------------------------------------------------------------------- */

/*
 * ABI 版本。任何**不向后兼容**的改动都必须 +1：
 *   - 增删/改结构体字段或顺序
 *   - 改函数签名、参数语义、返回值含义
 *   - 改数据布局（如 ka_tube 的字段排列）
 * 仅新增函数**不需要**升版本（旧调用方不受影响）。
 */
#define KA_ABI_VERSION 1

#if defined(_WIN32)
#  define KA_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define KA_API __attribute__((visibility("default")))
#else
#  define KA_API
#endif

/* ---------------------------------------------------------------------------
 * 状态码
 * ------------------------------------------------------------------------- */

typedef enum ka_status {
    KA_OK = 0,        /* 成功 */
    KA_ERR_NULL = 1,  /* 必需的指针为 NULL */
    KA_ERR_SIZE = 2,  /* 尺寸参数非法（负数、零、超出缓冲） */
    KA_ERR_GRID = 3,  /* 网格参数非法（grid_n <= 0 等） */
    KA_ERR_BOUNDS = 4,/* 包围盒非法（hi <= lo） */
    KA_ERR_ABI = 5,   /* ABI 版本不匹配 —— 调用方应停止而非继续 */
    KA_ERR_BUFFER = 6 /* 输出缓冲区容量不足（见 *_count 函数） */
} ka_status;

/* ---------------------------------------------------------------------------
 * 几何数据布局
 * ------------------------------------------------------------------------- */

/*
 * 一根管（δ-tube）。字段顺序与取值必须与 Python 侧
 * src/kakeya_ai/fast_kernel.py::tubes_to_array 的 (n_tubes, 8) 行**完全对应**：
 *
 *   索引:  0    1    2    3    4    5     6      7
 *   字段:  cx   cy   cz   dx   dy   dz  length  delta
 *
 * 该结构体本身**不用于跨 ABI 传递**（结构体对齐/padding 因平台而异）。
 * 传参一律用 double 数组、行优先（n_tubes * KA_TUBE_STRIDE 个 double），
 * 本结构体只是给调用方写数组时的字段索引文档。
 */
typedef struct ka_tube {
    double center[3];     /* 中心 */
    double direction[3];  /* 单位方向向量（调用方保证已归一化） */
    double length;        /* 轴长 */
    double delta;         /* 半径 */
} ka_tube;

/* 每根管在 double 数组里占多少格。用宏而非 sizeof(ka_tube)，
 * 以保证与 Python 侧 (n, 8) 的约定一致、不受 padding 影响。 */
#define KA_TUBE_STRIDE 8

/* bounds 数组的布局：3 个轴各一对 [lo, hi]，共 6 个 double，
 * 顺序与 Python 侧 bounds.reshape(3, 2) 一致：
 *   [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi] */
#define KA_BOUNDS_LEN 6

/* ---------------------------------------------------------------------------
 * 版本探测
 * ------------------------------------------------------------------------- */

/*
 * 返回本库的 ABI 版本。**调用方应在加载后先调它**，与自身编译时的
 * KA_ABI_VERSION 比对；不等就返回 KA_ERR_ABI，不要继续调用其它函数。
 *
 * 为什么不用"符号缺失"来判定：缺失的符号在部分平台上是惰性解析，
 * 拿到 NULL 函数指针后调用才是崩溃点，定位成本远高于一次显式查询。
 */
KA_API int32_t ka_abi_version(void);

/* 返回可读的库版本字符串（静态常量，调用方**不要** free）。
 * 例如 "kakeya-ai 0.1.0 abi=1"。可为 NULL 由调用方容错。 */
KA_API const char *ka_version_string(void);

/* ---------------------------------------------------------------------------
 * 1. 批量管-点包含测试
 * ------------------------------------------------------------------------- */

/*
 * 对 n_points 个点，返回每点是否被**至少一根**管包含。
 *
 * tubes      : n_tubes * KA_TUBE_STRIDE 个 double，行优先
 * n_tubes    : 管数，可为 0（此时全部点为 false）
 * points     : n_points * 3 个 double，行优先
 * n_points   : 点数，可为 0
 * out_mask   : 调用方提供的 n_points 字节缓冲区；写入 0 或 1
 *              （用 uint8_t 而非 bool：bool 的大小在 C/C++/其它语言间
 *               不保证一致，uint8_t 无歧义）
 *
 * 返回值：KA_OK，或 KA_ERR_NULL / KA_ERR_SIZE
 */
KA_API ka_status ka_contains_points_batch(
    const double *tubes, int32_t n_tubes,
    const double *points, int32_t n_points,
    uint8_t *out_mask);

/* ---------------------------------------------------------------------------
 * 2. 体素网格并集体积估计
 * ------------------------------------------------------------------------- */

/*
 * 用规则体素网格估计 |∪T|。体素中心落在任一管内即计为"内"。
 *
 * tubes    : 同 ka_contains_points_batch
 * n_tubes  : 管数
 * bounds   : KA_BOUNDS_LEN 个 double = [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]
 * grid_n   : 每轴体素数，必须 > 0。总体素数 = grid_n^3，
 *            调用方需自行估算内存/时间（grid_n=256 即 1.6e7 个体素）
 * out_volume: 写入体积估计
 *
 * 返回值：KA_OK，或 KA_ERR_NULL / KA_ERR_SIZE / KA_ERR_GRID / KA_ERR_BOUNDS
 *
 * 注意：本函数的时间复杂度是 O(grid_n^3 * n_tubes)，调用方负责选合适的规模。
 */
KA_API ka_status ka_voxel_union_volume(
    const double *tubes, int32_t n_tubes,
    const double *bounds, int32_t grid_n,
    double *out_volume);

/* ---------------------------------------------------------------------------
 * 3. 蒙特卡洛并集体积估计
 * ------------------------------------------------------------------------- */

/*
 * 用调用方预先给定的采样点做蒙特卡洛 |∪T| 估计。
 *
 * **为什么不在这里生成随机点**：随机数生成器的实现会显著影响可复现性，
 * 而本项目要求实验可复现（见 docs/PROJECT.md 第 IV 篇）。把采样交给调用方，
 * 本库只做确定性的"点在不在并集里"，结果可被独立复核。
 *
 * tubes       : 同 ka_contains_points_batch
 * n_tubes     : 管数
 * points      : n_samples * 3 个 double（调用方在包围盒内均匀采样）
 * n_samples   : 采样点数，必须 > 0
 * box_volume  : 采样所在盒的体积（用于把命中率换算成体积）
 * out_volume  : 写入估计值 = (命中数 / n_samples) * box_volume
 *
 * 返回值：KA_OK，或 KA_ERR_NULL / KA_ERR_SIZE
 */
KA_API ka_status ka_mc_union_volume(
    const double *tubes, int32_t n_tubes,
    const double *points, int32_t n_samples,
    double box_volume,
    double *out_volume);

/*
 * 蒙特卡洛并集体积（**融合版**，2026-10-30）——片内 splitmix64 采样，
 * 点数组不进内存（算子融合；能耗动机见 PROJECT.md 第 III 篇 §6）。
 * 与 ka_mc_union_volume 互补：那个消费调用方给的点，这个自己生成点。
 *
 * tubes       : (n_tubes, 8) double，布局同上
 * bounds      : 6 个 double，[x_lo,x_hi,y_lo,y_hi,z_lo,z_hi]（与
 *               ka_voxel_union_volume 同一布局）；box_volume 由
 *               (hi-lo) 连乘**内部得出**，无需调用方传
 * n_samples   : 采样点数，必须 > 0
 * seed        : 随机流种子——逐样本 RNG 只依赖 (seed, 样本序号)，
 *               同 seed 结果**逐位确定**（与 f32 pybind 层的融合 MC
 *               同一算法、同一随机流，精度口径 f64）
 * out_volume  : 写入估计值 = (命中数 / n_samples) * box_volume
 *
 * 返回值：KA_OK，或 KA_ERR_NULL / KA_ERR_SIZE / KA_ERR_BOUNDS
 * （仅新增函数，按本头文件约定不升 KA_ABI_VERSION）。
 */
KA_API ka_status ka_mc_union_volume_fused(
    const double *tubes, int32_t n_tubes,
    const double *bounds, int64_t n_samples, uint64_t seed,
    double *out_volume);

/* ---------------------------------------------------------------------------
 * 4. 网格单元 → 管 的分配（grains 构造热点）
 * ------------------------------------------------------------------------- */

/*
 * 把每根管的轴对齐包围盒分配到可能重叠的网格单元。
 *
 * 两段式调用（因为输出长度事先未知，而本库不分配内存）：
 *
 *   1) 先调 ka_assign_tubes_count() 拿到 out_count
 *   2) 调用方分配 2 * out_count 个 int32_t
 *   3) 再调 ka_assign_tubes() 填充
 *
 * bboxes   : n_tubes * 6 个 double，每管 [xlo,xhi,ylo,yhi,zlo,zhi]
 * n_tubes  : 管数
 * lo, hi   : 各 3 个 double，网格包围盒
 * nx,ny,nz : 各轴网格数，均必须 > 0
 * out_count: 写入 (cell_id, tube_index) 对的个数
 *
 * 返回值：KA_OK，或 KA_ERR_NULL / KA_ERR_SIZE / KA_ERR_GRID / KA_ERR_BOUNDS
 */
KA_API ka_status ka_assign_tubes_count(
    const double *bboxes, int32_t n_tubes,
    const double *lo, const double *hi,
    int32_t nx, int32_t ny, int32_t nz,
    int32_t *out_count);

/*
 * 填充上一步问出的容量。两个数组必须各能容纳 >= out_count 个元素。
 *
 * out_cell_ids : cell id，编码为 ix * ny * nz + iy * nz + iz
 *               （与 Python 侧 assign_tubes_to_grid_cells 一致）
 * out_tube_idx : 对应的管下标
 * capacity     : 两个数组各自的容量；若 < 实际所需，返回 KA_ERR_BUFFER
 *
 * 返回值：KA_OK，或 KA_ERR_NULL / KA_ERR_SIZE / KA_ERR_BUFFER
 */
KA_API ka_status ka_assign_tubes(
    const double *bboxes, int32_t n_tubes,
    const double *lo, const double *hi,
    int32_t nx, int32_t ny, int32_t nz,
    int32_t *out_cell_ids, int32_t *out_tube_idx, int32_t capacity);

/* ---------------------------------------------------------------------------
 * 状态码 → 可读说明（便于 CLI / 日志，不分配内存）
 * ------------------------------------------------------------------------- */

/* 返回静态字符串，调用方不要 free。未知码返回 "unknown status"。 */
KA_API const char *ka_status_string(ka_status status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KA_ABI_H */
