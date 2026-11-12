# drum-score-a4

把**滚动播放的鼓谱视频**（B 站 / YouTube 上的 Guitar-Pro 式滚动谱、打谱软件录制等）还原成
**干净的黑白、竖版 A4、可直接打印的鼓谱 PDF**。

图片式的"截图拼贴"不算鼓谱：这个小工具会把滚动视频拼成一条超长谱带，重建小节线，
再按排版规则重新排成 A4——小节不跨行、每行小节数固定、休止段用多小节休止号、
重复段落用 `×N` 标记、小节线统一加粗成一条黑线。

> Turn a scrolling drum-score video into a printable, portrait A4, black-on-white drum chart.

---

## 效果

| 输入 | 输出 |
| --- | --- |
| 竖屏录制的滚动谱视频（1080p），谱表横向逐段平移 | A4 竖版 PDF，每页 10~11 行，真白底 + 实黑音符 |

处理前后的差别：

- 背景由录制时的深色/照片/水印 → **纯白纸**，前景 → **纯黑墨**
- 播放光标、高亮、进度条 → 通过逐段中值合成消除
- 小节线 → 逐帧多行谱系统检测，避免把符干、谱号误当小节线
- 排版 → 固定 scale + 每个小节等宽槽位，小节永不被切断

## 流水线

```
视频 ──stitch.py──────────► *_strip.png        横向滚动的单行谱带
      （或 extract2d.py）    *_stitch.json      段偏移 / 拼接链接

视频 ──bars_from_frames.py─► *_framebars.json   逐帧、多谱系统的小节线 x 坐标

      map_bars.py ─────────► *_bars.json        映射到谱带坐标并去重

      a4layout.py ─────────► *.pdf              A4 排版输出（重排版）
      montage.py  ─────────► *.pdf              A4 拼贴输出（剪切粘贴，不漏音符）
```

`stitch.py` 和 `extract2d.py` 是两条互斥的取带路径：

- 谱面**只横向滚动** → 用 `stitch.py`，直接给固定 y 区间
- 谱面**同时有竖向位移**（画面里有多行谱、或镜头上下动） → 用 `extract2d.py`，
  以打击乐谱号（percussion clef）为锚点，逐段对齐到同一条鼓谱

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

依赖：`numpy`、`opencv-python-headless`、`pillow`。
下载视频请自备 `yt-dlp`（可选）；查看 PDF 页码可用 `pdfinfo`/`pdftoppm`（poppler，可选）。

## 用法

以下命令都在仓库根目录执行，`STEM` 是自己起的名字。

### 0) 取视频

```bash
yt-dlp -f "bv*[height<=1080]" -o src.mp4 "https://www.bilibili.com/video/BVxxxxxxxx"
```

### 1a) 横向滚动：拼谱带

```bash
python stitch.py src.mp4 300 470 out/stem     # 300/470 = 只含鼓谱那一行的 y 区间
```

先看几帧确定 y 区间：

```bash
ffmpeg -ss 10 -i src.mp4 -frames:v 1 out/frame.png
```

输出 `out/stem_strip.png`（可能几万像素宽）与 `out/stem_stitch.json`。
每个拼接链接都会打印 `corr` / `ink_mae`，`REJECT` 说明那一段没对齐，需要回头看。

> 拼接用的是**亮色墨迹掩膜**的列剖面（不是灰度）：背景里有动图/照片时，灰度剖面的残差
> 会被背景变化顶掉，而一条被 REJECT 的链接会**直接丢掉那一段音乐**（谱带只按接受的位移
> 前进）。所以只要谱是亮谱深底，就优先用掩膜判据。

### 1b) 谱面也有竖向位移：按谱号取带

```bash
python extract2d.py src.mp4 out/stem_drum.png --fixed-y 884 --skip-before 12
```

`--fixed-y` 直接用固定行（最稳，推荐先用它试）；不给则用谱号模板自动定位。
输出 `out/stem_drum.png` + 同名 `.json`。

### 2) 找小节线

```bash
python bars_from_frames.py src.mp4 out/stem_framebars.json
# 深色底、亮色谱 → 加 --bright
```

原理：一帧里有多行谱系统时，**真小节线会同时切穿多行谱**，而符干/谱号只属于一行。
所以要求某一列在全部谱系统里都是一条**连续**竖笔画（竖向形态学开运算，长度 ≥ 70% 谱高）。

局部反差（`g - GaussianBlur(g, 18) > 15`）会把被半透明照片压暗到中灰的小节线重新算回墨迹——
之前用绝对灰度带时，正是这里的漏检。

### 3) 映射到谱带坐标

```bash
python map_bars.py out/stem_stitch.json out/stem_framebars.json out/stem_bars.json
```

会打印小节间距统计，并丢掉间距远小于中位数的杂散线。

### 4) 排版成 A4

```bash
python a4layout.py out/stem_strip.png \
  --bars-json out/stem_bars.json \
  --clef-px 94,118 \
  --measures-per-line 6 \
  --page-numbers \
  -o out/score.pdf
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--mode dark\|bright` | 原视频是黑谱白底 / 亮谱深底 |
| `--ink` `--paper` | 重映射时"判定为实黑 / 纯白"的灰度值，微调对比度用 |
| `--staff-space-mm` | 谱线间距（毫米），决定整体大小，默认 1.55 |
| `--measures-per-line` | 每行固定小节数（0 = 按内容自动） |
| `--uniform` | 小节按**真实宽度**排，不横向拉伸（符头大小一致）；整行超宽时整行等比缩小，绝不叠压 |
| `--bar-width-pt` | 小节线粗细，默认 1.2pt（0 = 不画） |
| `--clef-px x0,x1` | 谱号模板在谱带中的像素范围，用于每行行首重绘谱号 |
| `--rest-ink` | 判定"空小节"的墨量阈值 |
| `--repeat-dist` | 相邻小节"同一音型"的剖面距离阈值（越小越严） |
| `--repeat-min` | 连续多少小节算一次重复组，默认 3 |
| `--phrase-len` | 以 N 小节为一个重复单位（如 4 → 4 小节音型 ×N） |
| `--collapse a-b:L` | 手工指定折叠：第 a..b 小节只印 L 小节，上方标 `×N` |
| `--highpass` `--gain` `--hp-floor` | 高通行清洗，去掉深底视频里的人像/照片背景 |
| `--debug` | 打印空小节区间、重复组、折叠区间，用来核对 |
| `--dpi` `--margin-mm` `--gap-mm` `--top-mm` | 版面细节 |

输出：`out/score.pdf` 以及每页的 `out/score.pdf.pNN.png`。

## 设计上的几个取舍

- **小节永不跨行**：换行只发生在小节线上，所以视口再窄也不会把小节切开。
- **等宽槽位**：每个小节占同样的宽度，行内再横向拉伸对齐；视觉上比"原始截图等比缩放"整齐。
- **一条加粗小节线**：拼接时相邻片段会重复出现同一条小节线，排版阶段只在小节起点画一条。
- **多小节休止**：连续的空白小节合并成一个粗横线 + 数字。
- **`×N` 折叠是启发式**：默认非常保守（只合并剖面几乎一致的小节）。
  拿不准时用 `--debug` 看它到底合并了哪几小节，必要时改用 `--collapse` 手工指定。
- **多小节休止占一小节位**：连续的空白小节合写成「一根细横条 + 上方数字」，而不是按空小节
  的总宽度占位（否则会把整行挤爆、把其他小节压小）。
- **首尾残块会被丢掉**：谱带最左的谱号区、视频结尾没滚完整的窄碎片都不算小节。
- **`--debug` 会打印每行的几何**（小节宽度、间隙、行宽），排版看起来不对时先看它。

## 多行谱（乐队总谱）

如果视频里是**多行谱叠在一起**（吉他/贝斯/鼓…），用 `system_layout.py`：整条谱带一起排版，
每页放若干套完整系统。

```bash
python stitch.py src.mp4 135 1000 out/band          # 把几行谱一起拼进来
python system_layout.py out/band_strip.png \
  --portrait --staff-space-mm 1.15 --systems-per-page 3 \
  --page-numbers -o out/band.pdf
```

它会自动：识别每行谱线 → 用「**所有行谱都被贯穿**」判定小节线 → 去掉视频里半透明的
播放光标带和人像/照片残留（中灰区间剔除）→ 按系统分页。

想只留其中一行（比如只要鼓），把谱带按 y 区间单独截出来，再交给 `a4layout.py` 即可。

## 只要鼓谱：从乐队总谱里切一行

源视频是**多行谱**（吉他 / 贝斯 / 鼓 叠在一起）而只要你那一行时，走这条线：

```bash
# 1) 整条谱带拼起来（小节线要横穿所有行才判得准）
python stitch.py src.mp4 135 1000 out/band

# 2) 重新取墨 + 找小节线（局部反差，照片压暗的线也找得到）
python - <<'EOF'
import cv2, json, numpy as np, system_layout as sl
ink, struct = sl.clean("out/band_strip.png", 35, 80, 160)
staves = sl.find_staves(struct)
bars = sl.bar_columns(ink, staves)
json.dump(bars, open("out/bars.json", "w"))
print(len(bars), "bar lines")
EOF

# 3) 只切鼓那一行，二值化成黑谱白纸
python cut_staff.py out/band_strip.png --bars out/bars.json \
  --rows 703:800 --lines 723,733,744,753,762 -o out/drum.png

# 4) 拼贴成 A4
python montage.py out/drum.png --bars out/bars.json \
  --lines 20,30,41,50,59 --per-line 4 -o out/drum.pdf
```

`cut_staff.py` 的 `--rows` 要**把谱线下方那几行连梁也包进来**（鼓谱的低音谱表会连梁
到谱线下方 30~35 px），`--lines` 给 5 条谱线的绝对行号；两者都可以先用
`find_staves()` 打出来看。

### 小节线漏检时

Guitar Pro 按音符数分宽度，所以同一份谱里 **16 分音符的小节会比 8 分音符的宽 ~1.5 倍**，
这是源的版面，不是漏检。如果确实有断开的线（照片盖住、或者源里那条线只画了半截），
用 `regularize.py` 按“本段的典型小节宽度”补齐：

```bash
python regularize.py out/band_strip.png --bars out/bars.json -o out/bars_fixed.json
# 88 -> 89 bars (+1); spacing median 265px min 127 max 406
```

它只补 **≥1.4 倍局部宽度**的空档（也就是“这一格明显能塞两小节”），并把插入的线吸附到
附近最像竖笔画的那一列（`--snap` 控制搜索范围）。

### `montage.py` 的三种宽度模式

| 模式 | 小节线 | 符头变形 | 适用 |
| --- | --- | --- | --- |
| `--cap 0.85`（默认） | 等密度的行严格等距（实测 20/23 行 1.00） | ≤15%（0.85–1.18） | 想要整齐又不想把音符压扁 |
| `--equal` | 全部等距 | 密集小节最窄压到 0.70 | 只要版面绝对整齐 |
| `--cap 0` | 跟随源的宽度（258–406 px） | 无 | 忠实还原视频版面 |

`--cap` 模式的原理：一条小节的横向缩放上下限是 `[cap*w, w/cap]`，整行可用宽度做
water-filling 求出**尽量相等**的槽位，剩下的宽度转成小节之间的空隙；一行全是密集小节
时其余放不下，就整行等比缩小（谱间距从 1.6mm 降到约 1.29mm），而不是把音符压得更扁。
一行里宽窄差太大（无法在 cap 内等分）时，该行的 cap 自动放宽到 `sqrt(min_w/max_w)`，
只影响这一行。

## 已知限制

- 假设画面里只有**一条**要提取的鼓谱；多行谱同时存在时需要先做竖向定位（`extract2d.py`）。
- 小节线检测依赖"多谱系统同时出现"，单行谱、或者视频被裁到只剩一行时，请把 `--bars-json`
  换成手工标注（就是一个 x 坐标的 JSON 数组）。
- 力度、装饰音、连谱号等记号的还原程度取决于源视频清晰度；这只是**重排版**，不是 OMR，
  不会识别音高/时值。
- 请只处理你有权使用的素材；谱面版权归原作者/出版方所有。

## 许可

MIT，见 [LICENSE](LICENSE)。
