/*
 * nano/isolated/bench_micro.cpp
 *
 * 纯 C++ 微基准：排除 Python/pybind11 开销，测 tube 几何核心在 M4 上的
 * **单核延迟**与**多核吞吐**，并把结果与芯片预算模型（nano.chip_model）
 * 的 roofline 上限对齐——这就是"纳秒内核 × 芯片调度/模型"的合流点
 * （2026-10-07 扩展）。
 *
 * 与 `cpp/kernel/nano_kernel.cpp` 的关系
 * ------------------------------
 * 本文件**自包含**（复制核心函数，不依赖 pybind11 / 包），因此能在**没有
 * Python 环境**的机器上直接编译运行——这是它作为"地面真值"的价值：Python
 * 层的计时会被 pybind11 与数据搬运稀释，这里不会。
 *
 * 并行口径与 `cpp/kernel/parallel.hpp` **一致**：按点区间切分、每个 worker 局部
 * int64 计数、最后按确定顺序求和（整数加法满足结合律 ⇒ 与串行逐位相同）。
 * 本文件不直接 include 那个头（要保持"无包依赖"），但**复制同一归约结构**，
 * 二者对同一构型应给出相同结果（见 tests/test_microbench_integration.py）。
 *
 * 异构分派 A/B（2026-10-30）：设 `KAKEYA_BENCH_PARTITION=1` 会追加一段
 * "even 均分 vs P/E 加权切分"的**同进程对比**——这是 PROJECT.md 第 III 篇
 * §6"按核类型分派"的**收益判据**（先判据后动手）。加权边界算法与
 * `cpp/kernel/parallel.hpp::weighted_bounds` **逐行同一口径**（刻意复制，保持无包
 * 依赖）；E 核权重常量见下 `E_CORE_WEIGHT`（三处同源，守门测试守）。
 *
 * 构建：
 *   clang++ -O3 -mcpu=apple-m4 -ffast-math -std=c++17 -pthread \
 *           bench_micro.cpp -o bench_micro
 *
 * 用 `KAKEYA_NUM_THREADS` 控制并行线程数（=1 只跑串行）。
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// ---------------------------------------------------------------------------
// 核心几何（与 nano_kernel.cpp 的 point_segment_dist2_f32 逐行一致）
// ---------------------------------------------------------------------------

inline float point_segment_dist2_f32(
    float px, float py, float pz,
    float ax, float ay, float az,
    float abx, float aby, float abz,
    float inv_ab_len2
) {
    float apx = px - ax;
    float apy = py - ay;
    float apz = pz - az;
    float t = (apx * abx + apy * aby + apz * abz) * inv_ab_len2;
    t = (t < 0.0f) ? 0.0f : ((t > 1.0f) ? 1.0f : t);
    float cx = ax + t * abx;
    float cy = ay + t * aby;
    float cz = az + t * abz;
    float dx = px - cx;
    float dy = py - cy;
    float dz = pz - cz;
    return dx * dx + dy * dy + dz * dz;
}

struct TubeData {
    float ax, ay, az;
    float abx, aby, abz;
    float inv_ab_len2;
    float delta_sq;
};

// ---------------------------------------------------------------------------
// 该构型下"每个 pair 的浮点运算数"——与 chip_model.py 的 FLOPS_PER_PAIR 同一
// 口径（25）。**故意在此重算一次**：若两处漂移，本基准打印的 roofline 就会
// 与模型不一致，等于自带交叉核对。
// ---------------------------------------------------------------------------
constexpr int FLOPS_PER_PAIR = 25;

// ---------------------------------------------------------------------------
// E 核**分派权重**（相对 P 核 = 1.0）——与 `cpp/kernel/parallel.hpp` 的
// `e_core_partition_weight()` **两处同源**（A/B 实测标定 = 0.60，见该头
// 文件注释里的标定实验数据；`KAKEYA_BENCH_PARTITION=1` 可复验）。
// **注意**：它不是 `nano/scheduler.py` 的 DEFAULT_E_CORE_EFFICIENCY（0.44）
// ——后者是 predict_schedule 速度模型参数（由均分曲线拟合），两者口径不同。
// 改这里必须同步 parallel.hpp——tests/test_microbench_integration.py 守。
// ---------------------------------------------------------------------------
constexpr double E_CORE_WEIGHT = 0.60;

// ---------------------------------------------------------------------------
// 融合 MC 片内采样（splitmix64，与 nano_kernel.cpp / geo_core.hpp 同一
// 算法——本文件按"无包依赖"惯例刻意复制；三处改算法时必须同改）
// ---------------------------------------------------------------------------
inline uint64_t splitmix64_next(uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline float unit_uniform_f32(uint64_t &state) {
    return static_cast<float>(
        (splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0));
}

//: 融合 MC **片内采样开销**（每样本 splitmix64×3 + 线性映射，ns）。
//: **标定结论（2026-10-30，配对交替协议，三连跑一致）**：开销在 ±1 ns
//: 测量噪声内（−0.9~−0.04 ns/sample），按 **0 计**——采样指令与管扫描
//: 相互独立，被乱序执行完全隐藏。"算力换带宽"的算力侧代价实测 ≈ 0。
//: 与 nano/chip_model.py 的 FUSED_MC_SAMPLING_OVERHEAD_NS **两处同源**，
//: tests/test_microbench_integration.py 守这个同步；改值必须重跑标定。
//: 标定两个已踩的坑（详见下方测量段注释）：纯函数累进局部变量会被
//: 提出循环（假快 145×）；背靠背整段计时被频率爬升倾斜（假差 ±2 ns）。
constexpr double FUSED_MC_SAMPLING_OVERHEAD_NS = 0.0;

// ---------------------------------------------------------------------------
// 芯片拓扑（与 cpp/kernel/parallel.hpp 同一套 sysctl 口径）
// ---------------------------------------------------------------------------

int _sysctl_int(const char *name) {
#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) return value;
#endif
    return 0;
}

struct Topology {
    int p_cores = 0;
    int e_cores = 0;
    int total_cores = 1;
    bool heterogeneous = false;
};

Topology detect_topology() {
    Topology t;
    t.p_cores = _sysctl_int("hw.perflevel0.logicalcpu");
    t.e_cores = _sysctl_int("hw.perflevel1.logicalcpu");
    t.total_cores = _sysctl_int("hw.ncpu");
    if (t.total_cores <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        t.total_cores = (hc == 0) ? 1 : static_cast<int>(hc);
    }
    if (t.p_cores <= 0 && t.e_cores <= 0) {
        t.p_cores = t.total_cores;
        t.e_cores = 0;
        t.heterogeneous = false;
    } else {
        t.heterogeneous = (t.p_cores > 0 && t.e_cores > 0);
    }
    return t;
}

int num_threads() {
    const char *env = std::getenv("KAKEYA_NUM_THREADS");
    if (env != nullptr) {
        char *end = nullptr;
        long v = std::strtol(env, &end, 10);
        if (end != env && v >= 1) return static_cast<int>(v);
    }
    return detect_topology().total_cores;
}

// ---------------------------------------------------------------------------
// 计数：串行 与 并行（与 cpp/kernel/parallel.hpp 同一归约结构）
// ---------------------------------------------------------------------------

//: 单个点是否被任一管包含（纯函数，供串行/并行共用）。
inline bool point_inside_any(
    const float *points, int j, const TubeData *tubes, int n_tubes
) {
    float px = points[j * 3 + 0];
    float py = points[j * 3 + 1];
    float pz = points[j * 3 + 2];
    for (int i = 0; i < n_tubes; ++i) {
        float d2 = point_segment_dist2_f32(
            px, py, pz,
            tubes[i].ax, tubes[i].ay, tubes[i].az,
            tubes[i].abx, tubes[i].aby, tubes[i].abz,
            tubes[i].inv_ab_len2
        );
        if (d2 <= tubes[i].delta_sq) return true;
    }
    return false;
}

int64_t count_serial(const float *points, int n_points, const TubeData *tubes, int n_tubes) {
    int64_t inside = 0;
    for (int j = 0; j < n_points; ++j) {
        if (point_inside_any(points, j, tubes, n_tubes)) ++inside;
    }
    return inside;
}

int64_t count_parallel(const float *points, int n_points, const TubeData *tubes,
                       int n_tubes, int n_threads) {
    if (n_threads <= 1) return count_serial(points, n_points, tubes, n_tubes);

    std::vector<int64_t> partials(static_cast<size_t>(n_threads), 0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_threads));

    const int64_t chunk = (static_cast<int64_t>(n_points) + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        const int64_t begin = static_cast<int64_t>(t) * chunk;
        const int64_t end = std::min<int64_t>(begin + chunk, n_points);
        if (begin >= end) break;
        workers.emplace_back([&, t, begin, end] {
            int64_t local = 0;
            for (int64_t j = begin; j < end; ++j) {
                if (point_inside_any(points, static_cast<int>(j), tubes, n_tubes)) ++local;
            }
            partials[static_cast<size_t>(t)] = local;
        });
    }
    for (auto &w : workers) w.join();

    int64_t total = 0;
    for (int64_t p : partials) total += p;
    return total;
}

//: 加权切分计数——**刻意复制** `cpp/kernel/parallel.hpp::weighted_bounds` 的边界算法
//: （保持无包依赖）：线程 0..p_cores-1 权重 1.0（P 核），其余权重 w_e（E 核），
//: 边界 = 前缀权重比例四舍五入、末边界钉死 n_points。区间不相交 ⇒ 计数仍与
//: 串行逐位相同。改动 parallel.hpp 时必须同步改这里（守门测试守权重常量，
//: 边界算法靠"计数逐位一致 + 文档声明"约束）。
int64_t count_parallel_weighted(const float *points, int n_points,
                                const TubeData *tubes, int n_tubes,
                                int n_threads, double w_e,
                                const Topology &topo) {
    if (n_threads <= 1) return count_serial(points, n_points, tubes, n_tubes);

    std::vector<double> w(static_cast<size_t>(n_threads), 1.0);
    for (int t = topo.p_cores; t < n_threads; ++t) {
        w[static_cast<size_t>(t)] = w_e;
    }
    double total = 0.0;
    for (double x : w) total += x;

    std::vector<int64_t> bounds(static_cast<size_t>(n_threads) + 1, 0);
    double cum = 0.0;
    for (int t = 0; t < n_threads; ++t) {
        cum += w[static_cast<size_t>(t)];
        bounds[static_cast<size_t>(t) + 1] =
            static_cast<int64_t>(std::llround(
                static_cast<double>(n_points) * cum / total));
    }
    bounds[static_cast<size_t>(n_threads)] = n_points;  // 钉死末边界

    std::vector<int64_t> partials(static_cast<size_t>(n_threads), 0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
        const int64_t begin = bounds[static_cast<size_t>(t)];
        const int64_t end = bounds[static_cast<size_t>(t) + 1];
        if (begin >= end) break;
        workers.emplace_back([&, t, begin, end] {
            int64_t local = 0;
            for (int64_t j = begin; j < end; ++j) {
                if (point_inside_any(points, static_cast<int>(j), tubes, n_tubes)) ++local;
            }
            partials[static_cast<size_t>(t)] = local;
        });
    }
    for (auto &wth : workers) wth.join();

    int64_t total_count = 0;
    for (int64_t p : partials) total_count += p;
    return total_count;
}

// ---------------------------------------------------------------------------
// 计时辅助
// ---------------------------------------------------------------------------

double time_count(int n_iter, const float *points, int n_points,
                  const TubeData *tubes, int n_tubes, int n_threads) {
    volatile int64_t sink = 0;
    // Warmup（1 轮，避免首次页错误/线程创建的噪声进结果）
    sink += count_parallel(points, n_points, tubes, n_tubes, n_threads);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < n_iter; ++it) {
        sink += count_parallel(points, n_points, tubes, n_tubes, n_threads);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;

    double per_call_s = std::chrono::duration<double>(t1 - t0).count() / n_iter;
    return per_call_s;  // 秒
}

int main(int argc, char **argv) {
    // 允许命令行覆盖构型：bench_micro [n_tubes] [n_points] [n_iter]
    int n_tubes = (argc > 1) ? std::atoi(argv[1]) : 64;
    int n_points = (argc > 2) ? std::atoi(argv[2]) : 100000;
    int n_iter = (argc > 3) ? std::atoi(argv[3]) : 200;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<TubeData> tubes(static_cast<size_t>(n_tubes));
    for (int i = 0; i < n_tubes; ++i) {
        float dx = dist(rng), dy = dist(rng), dz = dist(rng);
        float len = std::sqrt(dx * dx + dy * dy + dz * dz);
        dx /= len; dy /= len; dz /= len;
        float half = 0.5f;
        float ax = -half * dx, ay = -half * dy, az = -half * dz;
        float bx = half * dx, by = half * dy, bz = half * dz;
        float abx = bx - ax, aby = by - ay, abz = bz - az;
        float ab_len2 = abx * abx + aby * aby + abz * abz;
        tubes[i] = {ax, ay, az, abx, aby, abz, 1.0f / ab_len2, 0.08f * 0.08f};
    }

    std::vector<float> points(static_cast<size_t>(n_points) * 3);
    for (int j = 0; j < n_points; ++j) {
        float x = dist(rng), y = dist(rng), z = dist(rng);
        float r = std::sqrt(x * x + y * y + z * z);
        points[j * 3 + 0] = x / (2.0f * r);
        points[j * 3 + 1] = y / (2.0f * r);
        points[j * 3 + 2] = z / (2.0f * r);
    }

    Topology topo = detect_topology();
    int max_threads = num_threads();

    // 串行基准（1 线程）
    double serial_s = time_count(n_iter, points.data(), n_points,
                                 tubes.data(), n_tubes, 1);
    int64_t serial_count = count_serial(points.data(), n_points, tubes.data(), n_tubes);
    double pairs = static_cast<double>(n_tubes) * n_points;
    double serial_ns = serial_s * 1e9 / pairs;

    printf("Pure C++ micro-benchmark (no Python overhead)\n");
    printf("  Tubes: %d, Points: %d, Iterations: %d\n", n_tubes, n_points, n_iter);
    printf("  Topology: %dP + %dE = %d cores%s\n",
           topo.p_cores, topo.e_cores, topo.total_cores,
           topo.heterogeneous ? " (heterogeneous)" : "");
    printf("  Threads in use: %d%s\n", max_threads,
           max_threads == topo.total_cores ? " (all cores)" : " (overridden)");
    printf("  Serial count: %lld (prevents DCE)\n",
           static_cast<long long>(serial_count));
    printf("  Serial:  %.6f ms/call  |  %.4f ns/pair\n",
           serial_s * 1e3, serial_ns);

    // 并行扫描（若有多个核）
    if (max_threads > 1) {
        printf("\n  Parallel sweep (bit-identical count verified):\n");
        printf("    %-8s %12s %12s %8s %s\n",
               "threads", "ms/call", "ns/pair", "speedup", "count_match");
        for (int t : {2, 4, (topo.p_cores > 0 ? topo.p_cores : 4), max_threads}) {
            if (t <= 1 || t > max_threads) continue;
            double s = time_count(n_iter, points.data(), n_points,
                                  tubes.data(), n_tubes, t);
            int64_t c = count_parallel(points.data(), n_points, tubes.data(), n_tubes, t);
            bool match = (c == serial_count);
            printf("    %-8d %12.6f %12.4f %7.2fx %s\n",
                   t, s * 1e3, s * 1e9 / pairs, serial_s / s,
                   match ? "yes" : "**NO**");
            if (!match) {
                printf("  !! 并行计数 %lld != 串行 %lld —— 归约不逐位一致！\n",
                       static_cast<long long>(c), static_cast<long long>(serial_count));
                return 1;
            }
        }
    }

    // --- 异构分派 A/B（opt-in：KAKEYA_BENCH_PARTITION=1，2026-10-30）-------
    //
    // "按核类型分派"的收益判据（PROJECT.md 第 III 篇 §6：先判据后动手）。
    // 同进程、同数据、同线程数，只换切分方式：even（均分）vs P/E 加权
    // （w_e 取扫描值）——收益 = 加权 ns/pair 相对 even 的下降。每组都验
    // 计数逐位一致（加权不改变归约结果，这是前提不是优化）。
    const char *ab_env = std::getenv("KAKEYA_BENCH_PARTITION");
    if (ab_env != nullptr && ab_env[0] == '1' && max_threads > 1 &&
        topo.heterogeneous && max_threads > topo.p_cores) {
        printf("\n  Partition A/B (heterogeneous dispatch criterion):\n");
        printf("    %-14s %12s %12s %10s %s\n",
               "partition", "ms/call", "ns/pair", "vs even", "count_match");

        double even_s = time_count(n_iter, points.data(), n_points,
                                   tubes.data(), n_tubes, max_threads);
        int64_t even_c = count_parallel(points.data(), n_points, tubes.data(),
                                        n_tubes, max_threads);
        bool even_ok = (even_c == serial_count);
        printf("    %-14s %12.6f %12.4f %9s %s\n",
               "even", even_s * 1e3, even_s * 1e9 / pairs,
               "—", even_ok ? "yes" : "**NO**");
        bool all_ok = even_ok;

        for (double w_e : {0.30, 0.44, E_CORE_WEIGHT, 0.80}) {
            volatile int64_t sink = 0;
            // Warmup + 计时（与 time_count 同结构，但走加权路径）。
            sink += count_parallel_weighted(points.data(), n_points,
                                            tubes.data(), n_tubes,
                                            max_threads, w_e, topo);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < n_iter; ++it) {
                sink += count_parallel_weighted(points.data(), n_points,
                                                tubes.data(), n_tubes,
                                                max_threads, w_e, topo);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            (void)sink;
            double w_s = std::chrono::duration<double>(t1 - t0).count() / n_iter;
            int64_t wc = count_parallel_weighted(points.data(), n_points,
                                                 tubes.data(), n_tubes,
                                                 max_threads, w_e, topo);
            bool ok = (wc == serial_count);
            all_ok = all_ok && ok;
            char label[32];
            std::snprintf(label, sizeof(label), "weighted w=%.2f", w_e);
            printf("    %-14s %12.6f %12.4f %9.1f%% %s\n",
                   label, w_s * 1e3, w_s * 1e9 / pairs,
                   100.0 * (even_s - w_s) / even_s, ok ? "yes" : "**NO**");
        }
        if (!all_ok) {
            printf("  !! A/B 中出现计数不一致 —— 加权切分破坏了归约的逐位一致性！\n");
            return 1;
        }
    }

    // --- 融合 MC 片内采样开销（opt-in：KAKEYA_BENCH_FUSED_MC=1，2026-10-30）-
    //
    // 标定"算力换带宽"里**算力侧的代价**：splitmix64×3 + 线性映射的每样本
    // 开销（ns）。对照循环用整数散列代替 RNG（管判定成本两边相同），
    // 差值即纯采样开销。结果写入 FUSED_MC_SAMPLING_OVERHEAD_NS，与
    // nano/chip_model.py 两处同源（守门测试守）。
    const char *fused_mc_env = std::getenv("KAKEYA_BENCH_FUSED_MC");
    if (fused_mc_env != nullptr && fused_mc_env[0] == '1') {
        const int64_t n_samples = static_cast<int64_t>(n_points);
        volatile int64_t sink = 0;

        auto contains_xyz = [&](float px, float py, float pz) -> bool {
            for (int i = 0; i < n_tubes; ++i) {
                float d2 = point_segment_dist2_f32(
                    px, py, pz,
                    tubes[i].ax, tubes[i].ay, tubes[i].az,
                    tubes[i].abx, tubes[i].aby, tubes[i].abz,
                    tubes[i].inv_ab_len2);
                if (d2 <= tubes[i].delta_sq) return true;
            }
            return false;
        };

        // with_rng=true：融合口径（片内 splitmix64 采样）；
        // false：对照（整数散列坐标，无 RNG——隔离出纯采样开销）。
        auto one_pass = [&](bool with_rng) -> int64_t {
            int64_t inside = 0;
            for (int64_t j = 0; j < n_samples; ++j) {
                float px, py, pz;
                if (with_rng) {
                    uint64_t state = 42ULL +
                        0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(j + 1);
                    px = -1.0f + 2.0f * unit_uniform_f32(state);
                    py = -1.0f + 2.0f * unit_uniform_f32(state);
                    pz = -1.0f + 2.0f * unit_uniform_f32(state);
                } else {
                    const uint32_t h = static_cast<uint32_t>(j) * 2654435761u;
                    px = static_cast<float>(h % 2000u) * 0.001f - 1.0f;
                    py = static_cast<float>((h >> 11) % 2000u) * 0.001f - 1.0f;
                    pz = static_cast<float>((h >> 22) % 2000u) * 0.001f - 1.0f;
                }
                if (contains_xyz(px, py, pz)) ++inside;
            }
            return inside;
        };

        // **配对交替**采样：每轮 fused/control 各一遍、奇偶轮换先后——
        // 背靠背整段计时会被频率爬升/热状态倾斜（实测差值在 ±2 ns 间
        // 摆、甚至出现负值）；配对差让两种循环看见同一机器状态。
        // 每轮必须写 volatile sink：one_pass 是纯函数，累进局部变量会被
        // 编译器提出循环（初版实测差出 145× 就是踩了这个坑）。
        sink += one_pass(true);
        sink += one_pass(false);  // warmup
        double fused_total_s = 0.0, ctrl_total_s = 0.0, delta_total_s = 0.0;
        for (int it = 0; it < n_iter; ++it) {
            const bool fused_first = (it % 2 == 0);
            auto run_pair = [&](bool rng_first) {
                double &first_s = rng_first ? fused_total_s : ctrl_total_s;
                double &second_s = rng_first ? ctrl_total_s : fused_total_s;
                auto t0 = std::chrono::high_resolution_clock::now();
                sink += one_pass(rng_first);
                auto t1 = std::chrono::high_resolution_clock::now();
                sink += one_pass(!rng_first);
                auto t2 = std::chrono::high_resolution_clock::now();
                first_s += std::chrono::duration<double>(t1 - t0).count();
                second_s += std::chrono::duration<double>(t2 - t1).count();
                delta_total_s += std::chrono::duration<double>(t1 - t0).count()
                               - std::chrono::duration<double>(t2 - t1).count();
            };
            run_pair(fused_first);
        }
        const double dn = static_cast<double>(n_iter) *
                          static_cast<double>(n_samples);
        const double fused_ns = fused_total_s * 1e9 / dn;
        const double ctrl_ns = ctrl_total_s * 1e9 / dn;
        const double overhead_ns = delta_total_s * 1e9 / dn;
        const int64_t fused_cnt = one_pass(true);   // 打印用（不计时）
        const int64_t ctrl_cnt = one_pass(false);

        printf("\n  Fused-MC sampling overhead (splitmix64 x3 + map):\n");
        printf("    samples:      %lld (paired interleaved, %d iters)\n",
               static_cast<long long>(n_samples), n_iter);
        printf("    fused loop:   %.4f ns/sample (count=%lld)\n",
               fused_ns, static_cast<long long>(fused_cnt));
        printf("    control loop: %.4f ns/sample (count=%lld, no RNG)\n",
               ctrl_ns, static_cast<long long>(ctrl_cnt));
        printf("    overhead:     %.4f ns/sample (%.1f%% of fused)\n",
               overhead_ns, 100.0 * overhead_ns / fused_ns);
        printf("    sampling_overhead_ns_per_sample = %.4f\n", overhead_ns);
    }

    // --- 与芯片模型对齐的 roofline 对比（对**单核**口径）-------------------
    //
    // M4 参考峰值（与 chip_model.M4_REFERENCE **同源、同步维护**）：
    //   P 核：4.4 GHz × 4 fp32 通道 × 2(FMA) = 35.2 GFLOP/s/核；
    //   E 核：~P 的 30% ≈ 10.6 GFLOP/s/核；
    //   全芯片 = 4 × 35.2 + 6 × 10.6 ≈ 204 GFLOP/s。
    //
    // **单核对比必须用 P 核峰值**（实测跑在 P 核上；用"全芯片/核数"会得到
    // 偏低的每核峰值，使 measured/ceiling 虚高甚至 >100%，本基准初版就踩过）。
    const double p_core_gflops = 35.2;
    const double e_core_gflops = 10.6;
    const double chip_gflops_full =
        topo.p_cores * p_core_gflops + topo.e_cores * e_core_gflops;
    // 串行实测跑在单个核上——按 **P 核**峰值算上限（最贴近实际执行单元）。
    double single_core_ceiling_ns = FLOPS_PER_PAIR / (p_core_gflops * 1e9) * 1e9;

    printf("\n  Roofline (single-core, matches chip_model.M4_REFERENCE):\n");
    printf("    P-core peak:          %.1f GFLOP/s (float32, %.1f GHz x 4 x 2 FMA)\n",
           p_core_gflops, 4.4);
    printf("    chip peak (est.):     %.1f GFLOP/s (%dP + %dE)\n",
           chip_gflops_full, topo.p_cores, topo.e_cores);
    printf("    single-core ceiling:  %.4f ns/pair\n", single_core_ceiling_ns);
    double headroom = serial_ns / single_core_ceiling_ns;
    printf("    measured/ceiling:     %.2fx  (>=1.0 means at/near peak)\n", headroom);
    printf("    --> 单核已用 P 核峰值算力的 %.0f%%\n",
           100.0 * single_core_ceiling_ns / serial_ns);

    // --- 物理合理性门控（**依赖函数**，2026-10-07 用户定调）----------------
    // 与 Python 侧 chip_model.diagnose_superpeak 同一设计：不打印预写好的
    // "两种解释"，而是**真的去算**——
    //   ① 独立复算芯片峰值（clock × lanes × 2FMA × P + E×ratio），看声明值
    //      是否自洽；
    //   ② 由实测反推算力、再换算成"至少需要几个核"。
    // 然后**按计算结果**判定是哪一侧矛盾。实测快于上限即非零退出（2）。
    if (headroom < 1.0) {
        double implied_min_gflops = FLOPS_PER_PAIR / serial_ns;  // FLOP/ns = GFLOP/s

        // 检查①：独立复算——P 核峰值 = 4.4 × 4 × 2；E 核 = P × ratio。
        double recomputed_p_core = 4.4 * 4.0 * 2.0;             // 与 p_core_gflops 比
        double recomputed_full = topo.p_cores * recomputed_p_core
                               + topo.e_cores * recomputed_p_core
                                 * (e_core_gflops / p_core_gflops);
        double self_rel = (recomputed_full - chip_gflops_full) / recomputed_full;
        int self_ok = (self_rel < 0.0 ? -self_rel : self_rel) <= 0.15;

        // 检查②：反推需要几个核（单核以 p_core_gflops 为基准）。
        double cores_needed = implied_min_gflops / p_core_gflops;

        fprintf(stderr,
            "\n  !! 物理合理性门控：实测快于理论上限（headroom %.2fx < 1.0）\n"
            "     实测         : %.4f ns/pair\n"
            "     模型上限     : %.4f ns/pair\n"
            "     实测反推算力 : %.1f GFLOP/s (= %d FLOP/pair / 实测 ns)\n"
            "     反推需要核数 : %.2f 个（单核 %.1f GFLOP/s）\n"
            "     按计算得到的结论：\n"
            "       ① 芯片峰值%s（声明 %.1f vs 复算 %.1f GFLOP/s）\n"
            "       ② 实测反推需要 >=%.1f 个核的算力 => %s\n",
            headroom, serial_ns, single_core_ceiling_ns,
            implied_min_gflops, FLOPS_PER_PAIR,
            cores_needed, p_core_gflops,
            self_ok ? "自洽" : "**不自洽**", chip_gflops_full, recomputed_full,
            cores_needed,
            cores_needed > 1.05
                ? "这不是单线程实测（或单核峰值参数错）"
                : "若声称单线程，则单核峰值参数偏低");
        return 2;
    }
    return 0;
}
