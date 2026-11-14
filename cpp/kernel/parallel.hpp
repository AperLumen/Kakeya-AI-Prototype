/*
 * parallel.hpp — 把"整型计数"热点循环切到多核的最小工具（2026-10-07）。
 *
 * 为什么单独一个头文件：`nano_kernel.cpp` 与 `tube_kernel.cpp` 都要用它，
 * 而两份内核**本就可以独立编译**（`build_nano.sh` 逐个编译，不共享 TU）。
 * 头文件让"并行策略"只有一份实现——与 `src/kakeya_ai/cpp_loader.py` 收敛
 * 加载逻辑同一个思路（局限 #22）。
 *
 * 设计约束
 * --------
 * 1. **零新依赖**：只用 C++17 标准库 `<thread>`（不再是 OpenMP / TBB）。
 *    构建脚本需带 `-pthread`。
 * 2. **结果逐位一致**：本工具只服务"**整型**计数归约"——每个 worker 在
 *    自己的局部计数器上加，最后按**确定顺序**求和。整数加法满足结合律，
 *    故并行结果与串行**逐位相同**（浮点不满足，故浮点累加不走这里）。
 * 3. **可回退**：线程数可配（`KAKEYA_NUM_THREADS`），=1 时**完全走串行**
 *    路径，不创建线程。
 * 4. **不过度并行**：数据量小到"开线程比干活还贵"时，自动串行。
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace ka {

// ---------------------------------------------------------------------------
// 芯片拓扑感知（2026-10-07）
//
// **为什么需要**：M4 是**异构**芯片——4 个 Performance 核（快）+ 6 个
// Efficiency 核（慢、更省电）。此前默认线程数取 `min(hardware_concurrency, 8)`，
// 是**没依据的保守猜测**。实测（64 管，grid=128，M4）给出的加速曲线：
//
//   线程  1     2     3     4     5     6     7     8     10
//   ms   249.5 127.7 88.0  67.1  66.5  61.3  55.7  51.8  47.7
//
// 读法：**1→4（P 核）近似线性**（249→67，3.7×）；**4→10（E 核）仍在涨，
// 但斜率明显变缓**（67→48，再 1.4×）。⇒ 两条结论，都推翻了直觉：
//   1. **不能"只用到 P 核数"**——那会白丢 ~1.4×（E 核这活儿帮得上忙）；
//   2. **"上限 8"也是拍脑袋**——它丢掉了到 10 的那 ~8%。
//
// 所以默认改为"**按拓扑取全核**"，但仍**给内存守卫留口子**：本机约定
// GPU 只服务 LLM（`memory_guard`），当 Ollama 在跑时不该把 10 个核全占满
// 去和它抢内存带宽。这个"该用几个核"的判断**不在这里做**——本层只管
// "物理上有几个核、怎么分"；是否限流由上层按内存压力决定（传 KAKEYA_NUM_THREADS）。
// ---------------------------------------------------------------------------

//: 芯片拓扑：性能核与能效核的**逻辑** CPU 数（含超线程，若有）。
struct Topology {
    int p_cores = 0;      //: 性能核逻辑数（macOS 上的 perflevel0）
    int e_cores = 0;      //: 能效核逻辑数（macOS 上的 perflevel1）
    int total_cores = 0;  //: 上面之和；探测不到时 = hardware_concurrency
    bool heterogeneous = false;  //: 是否为异构芯片（P/E 都探到且不为 0）
};

#if defined(__APPLE__)
inline int _sysctl_int(const char *name) {
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) {
        return value;
    }
    return 0;
}
#endif

/** 探测本机核心拓扑。探测不到时退化为"全同构"，`total_cores` 取硬件并发度。 */
inline Topology detect_topology() {
    Topology topo;
#if defined(__APPLE__)
    // macOS：perflevel0 = 性能核，perflevel1 = 能效核。这是 Apple Silicon
    // 官方暴露的口径（`sysctl hw.perflevel0.logicalcpu`）。Intel Mac 上没有
    // 这两个键，取到 0 ⇒ 退化为同构。
    topo.p_cores = _sysctl_int("hw.perflevel0.logicalcpu");
    topo.e_cores = _sysctl_int("hw.perflevel1.logicalcpu");
    topo.total_cores = _sysctl_int("hw.ncpu");
#endif
    if (topo.total_cores <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        topo.total_cores = (hc == 0) ? 1 : static_cast<int>(hc);
    }
    if (topo.p_cores <= 0 && topo.e_cores <= 0) {
        // 同构（或探测失败）：把全部核当作"性能核"，行为与旧版一致。
        topo.p_cores = topo.total_cores;
        topo.e_cores = 0;
        topo.heterogeneous = false;
    } else {
        topo.heterogeneous = (topo.p_cores > 0 && topo.e_cores > 0);
    }
    return topo;
}

//: 拓扑探测结果（进程内探测一次，缓存）。
inline const Topology &topology() {
    static const Topology cached = detect_topology();
    return cached;
}

//: 允许的并行线程数。
//:
//: 优先级：**环境变量 `KAKEYA_NUM_THREADS` > 拓扑默认**。
//: 拓扑默认 = **全部逻辑核**（实测 P+E 一起用最快，见上文曲线）。
//: 上层若要给 Ollama 留核，设 `KAKEYA_NUM_THREADS`（例如留 2 个核跑
//: 推理，此处设 8）——**是否限流是上层的决定，不是本层的**。
inline int max_threads() {
    static const int cached = [] {
        const char *env = std::getenv("KAKEYA_NUM_THREADS");
        if (env != nullptr) {
            char *end = nullptr;
            long v = std::strtol(env, &end, 10);
            if (end != env && v >= 1) {
                return static_cast<int>(v);
            }
        }
        return topology().total_cores;
    }();
    return cached;
}

//: 低于这个工作量（= 单元数 × 每单元内层循环数）就不值得开线程。
//: 经验阈值：开一个线程约 20~50 µs，而一次内层几何计算约 1 ns/管对 ⇒
//: 单元数 × 管数 ≲ 几万时，串行更快。取保守值。
inline int64_t parallel_work_threshold() {
    return 200000;
}

//: 每线程完工统计（``KAKEYA_PARALLEL_STATS=1`` 开启，默认关、零开销）。
//: **用途**：工作窃取的**判据工具**（PROJECT.md 第 III 篇 §6——"更细的调度
//: 需要先有可判定的收益判据"）——先量"静态切分的失衡比 max/mean"，失衡不
//: 显著就不做动态调度（如实登记负结果，不加复杂度）。
inline bool parallel_stats_enabled() {
    static const bool cached = [] {
        const char *env = std::getenv("KAKEYA_PARALLEL_STATS");
        return env != nullptr && env[0] == '1';
    }();
    return cached;
}

//: 动态调度（"工作窃取"轻量版，2026-10-30）：各线程用**原子计数器**按需
//: 领取一小块活，领完为止——天然同时校正"静态权重标定误差"与"每单元
//: 成本 variance"两类失衡。环境变量 ``KAKEYA_CHUNK``：
//: **未设** = 自动（见 ``dynamic_units_threshold``：单元数少时用动态，
//: 单元数多时退回静态）；**0** = 强制静态（按 ``KAKEYA_PARTITION`` 加权/
//: 均分）；**N≥1** = 固定 chunk = N 单元。
//: **逐位一致不变**：原子领取的区间不相交，整数归约结果与切法无关。
//: 判据数据（2026-10-30 实测，见 PROJECT.md 第 III 篇 §6）：体素内核
//:（grid=128，仅 128 个粗单元、层间成本 variance 大）动态 chunk=2 比静态
//: 加权 **+7~8.5%**；contains 内核（30 万点、每点成本均匀）动态反而
//: **−2~3%**（原子开销 + 失局部性）⇒ "单元少→动态、单元多→静态"。
inline int64_t dynamic_chunk_env() {
    static const int64_t cached = [] {
        const char *env = std::getenv("KAKEYA_CHUNK");
        if (env == nullptr) return static_cast<int64_t>(-1);  // -1 = 自动
        char *end = nullptr;
        long v = std::strtol(env, &end, 10);
        if (end == env || v < 0) return static_cast<int64_t>(-1);
        return static_cast<int64_t>(v);
    }();
    return cached;
}

//: 单元数 ≤ 此阈值时用动态调度（自动模式下）。**标定依据**（2026-10-30）：
//: 128 单元（体素层）动态 +7~8.5%；300k 单元（逐点包含）动态 −2~3%。
//: 阈值取 8192——比体素常用网格（≤512 层）宽一个量级，又远低于逐点内核的
//: 典型单元数；两处实测点离它都远，误分类风险低。
inline int64_t dynamic_units_threshold() {
    return 8192;
}

// ---------------------------------------------------------------------------
// 异构分派（2026-10-30）：P/E 核**加权切分**工作量
// ---------------------------------------------------------------------------
//
// **为什么**：均分（``chunk = ceil(n/n_threads)``）默认每个线程干一样多的活，
// 但 M4 的 P 核与 E 核单线程吞吐量不同（分派权重见下，A/B 实测 0.60）⇒
// 异构芯片上**完工时间由最慢的 E 核线程决定**（它还在跑，P 核已空转）。
// 加权切分让每核分到的活**正比于其效率**——快慢核同时完工。
//
// **判据**（先判据后动手，PROJECT.md 第 III 篇 §6 登记；2026-10-30 实测，
// 两轮以上独立运行一致）：
// * **纯内核微基准**（``bench_micro``，``KAKEYA_BENCH_PARTITION=1``，64 管 ×
//   100k 点、数据驻留缓存、算力受限）：even 0.215~0.224 ns/pair；w=0.60 ⇒
//   **+4.4~4.8%**；w=0.80 ⇒ +3.8~4.6%；w=0.44 ⇒ −1.3~−2.3%；w=0.30 ⇒
//   −7.5~−8.1%。收益由真机扫描给出，不是拍脑袋。
// * **体素内核端到端**（``accel.volume_voxel``，64 管 grid=128，含数据准备
//   与 9 MB 点流访存）：加权与均分**差异在噪声内（±1%）**——该构型更接近
//   带宽受限、且逐格早退使每线程工作量本身不均，核间速率差被稀释。
// * **动态调度（工作窃取轻量版）**：静态切分的实测**失衡比 max/mean ≈
//   1.2~1.3**（``KAKEYA_PARALLEL_STATS=1`` 逐线程计时）。动态 chunk=2 在体素
//   内核比静态加权 **+7~8.5%**（两轮复现）；但在 contains 内核（30 万均匀
//   成本单元）反而 **−2~3%**（原子开销 + 失局部性）⇒ **单元少→动态、
//   单元多→静态**（阈值 8192，见 ``dynamic_units_threshold``）。
// 结论：默认 = 加权静态 + 少单元时动态（各场景均取实测胜者），
// ``KAKEYA_PARTITION=even`` / ``KAKEYA_CHUNK=0`` 可一行回退静态。
//
// **逐位一致不变**：与均分一样，各线程区间**不相交**、最后按 t 升序求和，
// 整数加法满足结合律 ⇒ 结果与串行逐位相同（边界怎么切都不影响计数）。
//
// **线程→核型的落位假设**：线程 0..p_cores-1 落在 P 核、其余在 E 核——
// 与 ``chip_model.predict_schedule`` 的"n 个线程先用 P 核、再用 E 核"同一
// 口径（macOS 不向用户暴露线程亲和性，只能依赖调度器通常行为；实测曲线
// 支持该假设）。线程数 ≤ P 核数时全是 P 核，加权退化为均分，不启用本路径。

//: E 核**分派权重**（相对 P 核 = 1.0），A/B 实测标定 = **0.60**。
//: **标定实验**（2026-10-30，``bench_micro`` 设 ``KAKEYA_BENCH_PARTITION=1``，
//: 64 管 × 100k 点 × 10 线程，两轮独立运行一致）：even 均分 0.2145~0.2236
//: ns/pair；w=0.30 → −7.8~−8.1%；w=0.44 → −1.3~−2.3%；**w=0.60 →
//: +4.4~+4.8%**；w=0.80 → +3.8%。故取 0.60。
//: **与 scheduler 的 DEFAULT_E_CORE_EFFICIENCY（0.44）不是一个量，勿混用**：
//: 后者是 ``predict_schedule`` 速度模型的参数，由**均分** (n,T(n)) 曲线拟合
//: （口径含均分失衡与争用）；前者是"各核同时完工"的**静态切分比例**，由 A/B
//: 直接标定。它也不是"E 核算力多强"（那是 ``ChipSpec.e_core_rate_ratio``
//: ~0.30）。
//: ⚠️ 改这里必须同步 ``nano/isolated/bench_micro.cpp`` 的 ``E_CORE_WEIGHT``
//: ——``tests/test_microbench_integration.py`` 守这个两处同源。
inline double e_core_partition_weight() {
    return 0.60;
}

//: 分派模式：**环境变量 ``KAKEYA_PARTITION``（"even" | "weighted"）> 默认**。
//: "weighted" 且拓扑异构、线程数超过 P 核数时，按 P/E 效率加权切分；
//: 其余情况（同构、线程 ≤ P 核数、显式 "even"）退回**均分**（与旧版逐行一致）。
//: 给 A/B 测量与异常排查留口子——默认行为改动可一行环境变量回退。
inline std::string partition_mode() {
    static const std::string cached = [] {
        const char *env = std::getenv("KAKEYA_PARTITION");
        if (env != nullptr) {
            const std::string v = env;
            if (v == "even" || v == "weighted") return v;
        }
        return std::string("weighted");
    }();
    return cached;
}

/**
 * 计算"按核效率加权"的切分边界（异构分派）。
 *
 * 线程 0..p_cores-1 权重 1.0（P 核），其余权重 ``e_core_partition_weight()``
 * （E 核）。第 t 个线程的边界按**前缀权重比例**四舍五入得出：
 * ``end_t = round(n_units × cum_w / total_w)``、``begin_{t+1} = end_t``。
 * 权重恒正 ⇒ 端点单调不减 ⇒ 区间既不重叠也不留缝；末边界钉死为
 * ``n_units``，防浮点累加的舍入漂移。
 *
 * 返回 ``n_threads + 1`` 个端点（``bounds[t]..bounds[t+1]`` 是线程 t 的区间）。
 */
inline std::vector<int64_t> weighted_bounds(int64_t n_units, int n_threads,
                                            const Topology &topo) {
    const double w_e = e_core_partition_weight();
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
                static_cast<double>(n_units) * cum / total));
    }
    bounds[static_cast<size_t>(n_threads)] = n_units;  // 钉死末边界
    return bounds;
}


/**
 * 把 `0..n_units` 切成若干块，每块算一个局部 int64 计数，求和返回。
 *
 * `fn(begin, end)` 必须是**纯函数**（只读共享数据、返回该区间的计数），
 * 且**不得写共享状态**——这是"逐位一致"的前提。
 */
template <typename Fn>
int64_t parallel_sum_units(int64_t n_units, int64_t work_per_unit, Fn fn) {
    const int n_threads = max_threads();
    const int64_t total_work = n_units * work_per_unit;

    // 串行回退：线程数=1，或工作量太小，或单元数太少（不值得切）。
    if (n_threads <= 1 || total_work < parallel_work_threshold() ||
        n_units < n_threads) {
        return fn(static_cast<int64_t>(0), n_units);
    }

    std::vector<int64_t> partials(static_cast<size_t>(n_threads), 0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n_threads));

    const bool stats = parallel_stats_enabled();
    std::vector<double> elapsed_ms(static_cast<size_t>(n_threads), 0.0);

    // 异构分派：仅在"加权模式 ∧ 异构拓扑 ∧ 线程数超过 P 核数"时启用——
    // 其余情况（含所有同构机器）走与旧版逐行一致的均分路径，行为不变。
    const Topology &topo = topology();
    const bool weighted =
        topo.heterogeneous && partition_mode() == "weighted" &&
        n_threads > topo.p_cores;
    std::vector<int64_t> bounds;
    if (weighted) bounds = weighted_bounds(n_units, n_threads, topo);

    // 动态调度：原子计数器按需领活，替代静态切分。
    // 尺寸口径：显式 N > 用 N；未设（自动）且单元数 ≤ 阈值 ⇒ 自适应
    // chunk ≈ 单元数/(线程数×6)（每线程约领 6 块，粒度与单元总量挂钩）；
    // =0 或单元数超阈值 ⇒ 静态。
    const int64_t env_chunk = dynamic_chunk_env();
    int64_t dyn_chunk = 0;
    if (env_chunk > 0) {
        dyn_chunk = env_chunk;
    } else if (env_chunk < 0 && n_units <= dynamic_units_threshold()) {
        dyn_chunk = std::max<int64_t>(
            1, (n_units + static_cast<int64_t>(n_threads) * 6 - 1) /
                   (static_cast<int64_t>(n_threads) * 6));
    }
    std::atomic<int64_t> next_unit{0};

    const int64_t chunk = (n_units + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        if (dyn_chunk > 0 && n_units >= 2 * dyn_chunk) {
            workers.emplace_back([&, t] {
                for (;;) {
                    const int64_t begin =
                        next_unit.fetch_add(dyn_chunk, std::memory_order_relaxed);
                    if (begin >= n_units) break;
                    const int64_t end =
                        std::min<int64_t>(begin + dyn_chunk, n_units);
                    const auto w0 = std::chrono::steady_clock::now();
                    partials[static_cast<size_t>(t)] += fn(begin, end);
                    if (stats) {
                        elapsed_ms[static_cast<size_t>(t)] +=
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - w0).count();
                    }
                }
            });
            continue;
        }
        int64_t begin, end;
        if (weighted) {
            begin = bounds[static_cast<size_t>(t)];
            end = bounds[static_cast<size_t>(t) + 1];
        } else {
            begin = static_cast<int64_t>(t) * chunk;
            end = std::min<int64_t>(begin + chunk, n_units);
        }
        if (begin >= end) break;  // 块已分配完（n_units 不是整数倍时；权重恒正
                                  // ⇒ 端点单调，空块之后必全空，break 安全）
        workers.emplace_back([&, t, begin, end] {
            const auto w0 = std::chrono::steady_clock::now();
            partials[static_cast<size_t>(t)] = fn(begin, end);
            if (stats) {
                elapsed_ms[static_cast<size_t>(t)] =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - w0).count();
            }
        });
    }
    for (auto &w : workers) w.join();

    if (stats) {
        // 失衡比 = max/mean：≈1.0 表示各线程同时完工（静态切分已够）；
        // 明显 >1 才有动态调度（工作窃取）的收益空间。
        double sum = 0.0;
        int active = 0;
        for (size_t t = 0; t < elapsed_ms.size(); ++t) {
            if (partials[t] > 0 || elapsed_ms[t] > 0.0) {
                sum += elapsed_ms[t];
                ++active;
            }
        }
        if (active > 0) {
            const double mean = sum / active;
            double max_v = 0.0;
            for (double e : elapsed_ms) max_v = std::max(max_v, e);
            std::fprintf(stderr,
                         "[parallel_stats] threads=%d active=%d mean=%.3fms "
                         "max=%.3fms imbalance=%.3f\n",
                         n_threads, active, mean, max_v, mean > 0 ? max_v / mean : 0.0);
        }
    }

    // **按 t 升序**求和：整数加法结合律保证与串行逐位相同，且顺序确定。
    int64_t total = 0;
    for (int64_t p : partials) total += p;
    return total;
}

}  // namespace ka
