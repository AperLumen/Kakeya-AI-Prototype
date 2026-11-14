/*
 * ka_cli.cpp — 无 Python 依赖的命令行入口
 *
 * 这是"通用性可验证"的落点：在一台**没有 Python** 的机器上 `make` 出本程序，
 * 喂它一组 CSV，就能得到体积估计。跑通即证明这层接口不绑定 Python 生态。
 *
 * 用法：
 *   ka abi-version
 *   ka contains  --tubes T.csv --points P.csv
 *   ka volume    --tubes T.csv --grid-n 128 [--bounds -1,1,-1,1,-1,1]
 *   ka mc        --tubes T.csv --points P.csv --box-volume 8.0
 *   ka mc-fused  --tubes T.csv --samples N [--seed S] [--bounds ...]
 *                  （融合版：片内 splitmix64 采样，点数组不进内存；
 *                    同 seed 结果逐位确定，2026-10-30）
 *   ka assign    --bboxes B.csv --grid 16,16,16 [--bounds ...]
 *
 * CSV 格式（无表头，逗号分隔，允许空行与 '#' 注释行）：
 *   --tubes   : 8 列 = cx,cy,cz,dx,dy,dz,length,delta
 *   --points  : 3 列 = x,y,z
 *   --bboxes  : 6 列 = xlo,xhi,ylo,yhi,zlo,zhi
 *
 * 输出：结果写 stdout（纯数值或 CSV），诊断写 stderr。
 * 退出码：0 成功；1 参数/IO 错误；2 库返回错误。
 *
 * 为什么自己写 CSV 解析而不用库：本文件的目标是"零第三方依赖、能在
 * 最小环境编译"。CSV 子集（逗号分隔 double）用不到 30 行。
 */

#include "ka_abi.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *kUsage =
    "ka — Kakeya AI 几何内核（无 Python 依赖的 C ABI 前端）\n"
    "\n"
    "用法：\n"
    "  ka abi-version\n"
    "  ka contains --tubes T.csv --points P.csv\n"
    "  ka volume   --tubes T.csv --grid-n N [--bounds lo,hi,lo,hi,lo,hi]\n"
    "  ka mc       --tubes T.csv --points P.csv --box-volume V\n"
    "  ka mc-fused --tubes T.csv --samples N [--seed S] [--bounds lo,hi,...]\n"
    "              （片内 splitmix64 采样，点数组不进内存；同 seed 逐位确定）\n"
    "  ka assign   --bboxes B.csv --grid nx,ny,nz [--bounds lo,hi,...]\n"
    "\n"
    "CSV 列数：tubes=8（cx,cy,cz,dx,dy,dz,length,delta）\n"
    "         points=3（x,y,z）   bboxes=6（xlo,xhi,ylo,yhi,zlo,zhi）\n"
    "默认 bounds = -1,1,-1,1,-1,1\n";

/* 读 CSV：每行按 ',' 切分成 double。跳过空行与 '#' 开头。
 * expect_cols > 0 时校验列数，不符即报错（宁可显式失败，不静默错位）。 */
bool read_csv(const std::string &path, int expect_cols,
              std::vector<double> &out, std::string &err) {
    std::ifstream fh(path);
    if (!fh) {
        err = "无法打开文件：" + path;
        return false;
    }
    std::string line;
    int64_t lineno = 0;
    int64_t nrows = 0;
    while (std::getline(fh, line)) {
        ++lineno;
        /* 去掉行尾 \r（Windows 生成的 CSV）与首尾空白 */
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        size_t b = line.find_first_not_of(" \t");
        if (b == std::string::npos) continue;          /* 空行 */
        if (line[b] == '#') continue;                  /* 注释 */
        line = line.substr(b);

        std::stringstream ss(line);
        std::string cell;
        int cols = 0;
        while (std::getline(ss, cell, ',')) {
            ++cols;
            /* 空单元格视为 0（常见于尾部多余逗号） */
            if (cell.find_first_not_of(" \t") == std::string::npos) {
                out.push_back(0.0);
                continue;
            }
            char *end = nullptr;
            errno = 0;
            const double v = std::strtod(cell.c_str(), &end);
            if (end == cell.c_str() || *end != '\0' || errno == ERANGE) {
                err = "第 " + std::to_string(lineno) + " 行第 " +
                      std::to_string(cols) + " 列不是合法数字：'" + cell + "'";
                return false;
            }
            out.push_back(v);
        }
        if (expect_cols > 0 && cols != expect_cols) {
            err = "第 " + std::to_string(lineno) + " 行有 " +
                  std::to_string(cols) + " 列，期望 " + std::to_string(expect_cols);
            return false;
        }
        ++nrows;
    }
    if (nrows == 0) {
        err = "文件为空或全是注释：" + path;
        return false;
    }
    return true;
}

/* 解析含逗号的 double 列表，如 "-1,1,-1,1,-1,1" 或 "16,16,16"。 */
bool parse_doubles(const std::string &s, std::vector<double> &out,
                   std::string &err) {
    std::stringstream ss(s);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        char *end = nullptr;
        const double v = std::strtod(cell.c_str(), &end);
        if (end == cell.c_str() || *end != '\0') {
            err = "不是合法的数字列表：'" + s + "'";
            return false;
        }
        out.push_back(v);
    }
    return true;
}

bool parse_int(const std::string &s, int32_t &out) {
    char *end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v <= 0 || v > 100000) return false;
    out = static_cast<int32_t>(v);
    return true;
}

/* 取 --key value 形式的参数；找不到返回空。 */
bool opt(const std::vector<std::string> &args, const std::string &key,
         std::string &out) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            out = args[i + 1];
            return true;
        }
    }
    return false;
}

int fail(const std::string &msg) {
    std::fprintf(stderr, "ka: %s\n", msg.c_str());
    return 1;
}

int lib_fail(const char *what, ka_status st) {
    std::fprintf(stderr, "ka: %s 失败：%s\n", what, ka_status_string(st));
    return 2;
}

/* 包围盒：默认 [-1,1]^3，与 Python 侧 fast_kernel 的默认一致。 */
void default_bounds(double *b) {
    b[0] = -1.0; b[1] = 1.0;
    b[2] = -1.0; b[3] = 1.0;
    b[4] = -1.0; b[5] = 1.0;
}

} /* namespace */

int main(int argc, char **argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help") {
        std::fputs(kUsage, stdout);
        return args.empty() ? 1 : 0;
    }

    const std::string cmd = args[0];

    /* 版本探测：调用方（任意语言）应当先做这件事 */
    if (cmd == "abi-version") {
        std::printf("%d\n", ka_abi_version());
        return 0;
    }
    if (cmd == "version") {
        std::printf("%s\n", ka_version_string() ? ka_version_string() : "(null)");
        return 0;
    }

    double bounds[KA_BOUNDS_LEN];
    default_bounds(bounds);
    std::string s;
    if (opt(args, "--bounds", s)) {
        std::vector<double> v;
        std::string err;
        if (!parse_doubles(s, v, err) || v.size() != KA_BOUNDS_LEN)
            return fail("--bounds 需要 6 个数字（lo,hi × 3 轴）");
        for (int i = 0; i < KA_BOUNDS_LEN; ++i) bounds[i] = v[i];
    }

    if (cmd == "contains") {
        if (!opt(args, "--tubes", s)) return fail("缺少 --tubes");
        std::vector<double> tubes, points;
        std::string err;
        if (!read_csv(s, KA_TUBE_STRIDE, tubes, err)) return fail(err);
        if (!opt(args, "--points", s)) return fail("缺少 --points");
        if (!read_csv(s, 3, points, err)) return fail(err);

        const int32_t n_tubes = static_cast<int32_t>(tubes.size() / KA_TUBE_STRIDE);
        const int32_t n_points = static_cast<int32_t>(points.size() / 3);
        std::vector<uint8_t> mask(static_cast<size_t>(n_points));
        const ka_status st = ka_contains_points_batch(
            tubes.data(), n_tubes, points.data(), n_points, mask.data());
        if (st != KA_OK) return lib_fail("ka_contains_points_batch", st);

        /* CSV 输出：index,inside —— 便于任何语言/Excel 直接消费 */
        std::printf("index,inside\n");
        for (int32_t j = 0; j < n_points; ++j)
            std::printf("%d,%d\n", j, mask[j] ? 1 : 0);
        return 0;
    }

    if (cmd == "volume") {
        if (!opt(args, "--tubes", s)) return fail("缺少 --tubes");
        std::vector<double> tubes;
        std::string err;
        if (!read_csv(s, KA_TUBE_STRIDE, tubes, err)) return fail(err);
        int32_t grid_n = 0;
        if (!opt(args, "--grid-n", s) || !parse_int(s, grid_n))
            return fail("--grid-n 需要正整数");

        const int32_t n_tubes = static_cast<int32_t>(tubes.size() / KA_TUBE_STRIDE);
        double vol = 0.0;
        const ka_status st = ka_voxel_union_volume(
            tubes.data(), n_tubes, bounds, grid_n, &vol);
        if (st != KA_OK) return lib_fail("ka_voxel_union_volume", st);

        std::printf("volume\n%.17g\n", vol);
        return 0;
    }

    if (cmd == "mc") {
        if (!opt(args, "--tubes", s)) return fail("缺少 --tubes");
        std::vector<double> tubes, points;
        std::string err;
        if (!read_csv(s, KA_TUBE_STRIDE, tubes, err)) return fail(err);
        if (!opt(args, "--points", s)) return fail("缺少 --points");
        if (!read_csv(s, 3, points, err)) return fail(err);
        if (!opt(args, "--box-volume", s)) return fail("缺少 --box-volume");
        std::vector<double> bv;
        if (!parse_doubles(s, bv, err) || bv.size() != 1)
            return fail("--box-volume 需要 1 个数字");

        const int32_t n_tubes = static_cast<int32_t>(tubes.size() / KA_TUBE_STRIDE);
        const int32_t n_samples = static_cast<int32_t>(points.size() / 3);
        double vol = 0.0;
        const ka_status st = ka_mc_union_volume(
            tubes.data(), n_tubes, points.data(), n_samples, bv[0], &vol);
        if (st != KA_OK) return lib_fail("ka_mc_union_volume", st);

        std::printf("volume\n%.17g\n", vol);
        return 0;
    }

    if (cmd == "mc-fused") {
        if (!opt(args, "--tubes", s)) return fail("缺少 --tubes");
        std::vector<double> tubes;
        std::string err;
        if (!read_csv(s, KA_TUBE_STRIDE, tubes, err)) return fail(err);
        if (!opt(args, "--samples", s)) return fail("缺少 --samples");
        char *end = nullptr;
        const long long n_ll = std::strtoll(s.c_str(), &end, 10);
        if (end == s.c_str() || n_ll <= 0) return fail("--samples 需要正整数");
        uint64_t seed = 0;
        if (opt(args, "--seed", s)) {
            end = nullptr;
            const unsigned long long s_ull = std::strtoull(s.c_str(), &end, 10);
            if (end == s.c_str()) return fail("--seed 需要非负整数");
            seed = static_cast<uint64_t>(s_ull);
        }

        const int32_t n_tubes = static_cast<int32_t>(tubes.size() / KA_TUBE_STRIDE);
        double vol = 0.0;
        const ka_status st = ka_mc_union_volume_fused(
            tubes.data(), n_tubes, bounds, n_ll, seed, &vol);
        if (st != KA_OK) return lib_fail("ka_mc_union_volume_fused", st);

        std::printf("volume\n%.17g\n", vol);
        return 0;
    }

    if (cmd == "assign") {
        if (!opt(args, "--bboxes", s)) return fail("缺少 --bboxes");
        std::vector<double> bboxes;
        std::string err;
        if (!read_csv(s, 6, bboxes, err)) return fail(err);
        if (!opt(args, "--grid", s)) return fail("缺少 --grid");
        std::vector<double> grid;
        if (!parse_doubles(s, grid, err) || grid.size() != 3)
            return fail("--grid 需要 3 个分量 nx,ny,nz");
        int32_t nx = 0, ny = 0, nz = 0;
        const std::string gx = std::to_string(static_cast<long>(grid[0]));
        const std::string gy = std::to_string(static_cast<long>(grid[1]));
        const std::string gz = std::to_string(static_cast<long>(grid[2]));
        if (!parse_int(gx, nx) || !parse_int(gy, ny) || !parse_int(gz, nz))
            return fail("--grid 分量必须是正整数");

        const double lo[3] = {bounds[0], bounds[2], bounds[4]};
        const double hi[3] = {bounds[1], bounds[3], bounds[5]};
        const int32_t n_tubes = static_cast<int32_t>(bboxes.size() / 6);

        int32_t count = 0;
        ka_status st = ka_assign_tubes_count(
            bboxes.data(), n_tubes, lo, hi, nx, ny, nz, &count);
        if (st != KA_OK) return lib_fail("ka_assign_tubes_count", st);

        std::vector<int32_t> cell_ids(static_cast<size_t>(count));
        std::vector<int32_t> tube_idx(static_cast<size_t>(count));
        st = ka_assign_tubes(bboxes.data(), n_tubes, lo, hi, nx, ny, nz,
                             cell_ids.data(), tube_idx.data(), count);
        if (st != KA_OK) return lib_fail("ka_assign_tubes", st);

        std::printf("cell_id,tube_index\n");
        for (int32_t i = 0; i < count; ++i)
            std::printf("%d,%d\n", cell_ids[i], tube_idx[i]);
        return 0;
    }

    std::fprintf(stderr, "ka: 未知子命令 '%s'（用 --help 看用法）\n", cmd.c_str());
    return 1;
}
