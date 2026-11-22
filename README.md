<p align="center"><img src="docs/icon.png" width="112" alt="SensorLens"></p>

# SensorLens 感知透镜

智能驾驶多传感器标定与标注的可视化工具。给它图片和配对的 JSON（内外参、位姿、标注，点云可选），它会自动识别 JSON 格式、完成坐标系转换并校验标定，把点云、3D 框、车道线、关键点和 2D 标注画到图像上，也能生成 BEV 俯视图、3D 视图和多帧拼接，结果可以一次导出到指定文件夹。

只有“图片 + JSON”（没有点云，甚至没有内外参）的数据也能用：2D 标注直接画在图上，有内外参时 3D 标注也会投影上去。

## 源码运行

需要 Python 3.10 及以上。

```bash
git clone https://github.com/Apageoflove/SensorLens.git
cd SensorLens
python -m pip install -r requirements.txt
python -m uvicorn sensorlens.server:app --port 8765     # 浏览器打开 http://127.0.0.1:8765
```

Windows 上也可以用脚本：

```powershell
.\run.ps1            # 启动服务，打开 http://127.0.0.1:8765
.\test.ps1           # 单元测试（需要 pip install -r requirements-build.txt）
```

- 脚本默认使用仓库里的 `.venv`（存在时）或 PATH 上的 `python`。要指定解释器，设置环境变量 `SENSORLENS_PYTHON`，或者在仓库根目录建一个 `env.local.ps1` 写入 `$env:SENSORLENS_PYTHON = "<python.exe 的路径>"`（这个文件不会提交）。
- `env.ps1` 把临时文件和 Python 字节码缓存放在仓库内的 `.cache/`，上传的数据放在 `data/sessions/`。
- 服务只监听 127.0.0.1。

先用演示数据试一下：

| 样例 | 内容 |
|---|---|
| `samples/demo.zip`（或 `samples/demo` 文件夹） | 5 帧，两个相机（含鱼眼）+ 旋转雷达（带逐点时间），3D 框、2D 框、车道线、行人脚点 |
| `samples/vendor_demo` | 第三方格式（格式选 `vendor_demo`），binary_compressed PCD，微秒时间戳 |
| `samples/image_only` | 只有一张图片 + 2D 框 / 2D 车道线，没有内外参和点云 |
| `samples/mono3d` | 图片 + 内外参 + 2D/3D 框，没有点云 |

重新生成演示数据：

```bash
python -m sensorlens.cli demo samples --frames 5
```

## 桌面版（Windows / Linux）

到 [Releases](https://github.com/Apageoflove/SensorLens/releases/latest) 下载：

| 文件 | 说明 |
|---|---|
| `SensorLens_<版本>_x64-setup.exe` | Windows 安装版（装到当前用户，不需要管理员权限） |
| `SensorLens_<版本>_amd64.deb` / `.rpm` | Ubuntu、Debian / Fedora、openSUSE 安装版 |
| `SensorLens_<版本>_amd64.AppImage` | 任意 Linux 发行版，`chmod +x` 后直接运行 |
| `*_portable.zip` / `*_portable.tar.gz` | 便携版，解压即用，不写系统目录 |

- 第一次打开时选择数据目录（上传的数据、索引缓存、日志都放在这里）。之后可以点右下角“数据目录”切换，旧目录不会被删除。
- 安装版点右下角“检查更新”即可下载并覆盖安装新版本，数据目录和设置保留不变。便携版会打开下载页，下载新版后把文件夹替换掉即可（数据目录 `SensorLens-data` 留着；从旧名 ad-viz 升级时沿用原来的 `ad-viz-data`）。
- Linux 需要 WebKitGTK 4.1（Ubuntu 22.04+ 自带）。
- 后端只监听 127.0.0.1 的随机端口，并要求一次性令牌，本机其他程序和网页访问不到。

### 从源码构建

```powershell
python -m venv .venv-build; .venv-build\Scripts\pip install -r requirements-build.txt
.venv-build\Scripts\python packaging\build_backend.py      # PyInstaller 打包后端到 build\backend
cd desktop\src-tauri; cargo tauri build --bundles nsis      # Linux: --bundles deb,rpm,appimage
```

发布流程：改 `desktop/src-tauri/tauri.conf.json` 和 `Cargo.toml` 里的版本号，推送 `v<版本>` 标签。GitHub Actions 会先跑测试，再在 Windows 和 Ubuntu 上构建，签名后发布到 Releases，并生成更新器读取的 `latest.json`。签名私钥保存在仓库 Secret `TAURI_SIGNING_PRIVATE_KEY`，公钥写在 `tauri.conf.json`；私钥丢失后旧版本将无法自动更新。

## 页面功能

只需提供图片和配对的 JSON，其余自动完成：

- 自动选择可视化：根据数据里实际有什么（相机数、内外参、点云、位姿、2D/3D 标注）决定显示哪些视图。多相机默认进入环视，有点云或 3D 标注时提供 BEV 和 3D 视图，有位姿的多帧点云可以拼接。侧栏会列出自动选择的视图。
- 图片完整性：每张图片在叠加投影前都会检查。数据里有同名的 zip 副本时（如 nuScenes 目录里的 `samples.zip`）比对 CRC，磁盘文件损坏就自动改用副本；没有副本时检查 JPEG 结束标记并完整解码一次，损坏的图片不出图、不叠加投影，并说明原因。拷贝或解压中断造成的半张灰图不会再被当成正常图片。
- 自动标定校验 / 修正：每个序列的每个相机在打开时取 4 帧，对外参方向、四元数顺序、相机轴约定的各种组合打分（方法见“外参检查原理”）。多数帧都认为另一种约定明显更好时自动改用；有点云或 2D 框可作依据时，再在 ±3° 内对相机姿态做小角度精调，只在多数帧都变好时采用。结论（已验证 / 无法自动验证 / 不吻合）显示在输出栏。不吻合又修不好的相机只显示 2D 标注，不画可能误导的 3D 投影。
- 导出到文件夹：侧栏“导出可视化”选择目标文件夹（桌面版可点“浏览…”）和范围（当前帧、当前序列、全部帧或自定义帧），一次写出每个相机的投影图、环视拼图、BEV 俯视图，以及 `report.json` / `report.csv`（每个文件的状态、标定结论、跳过的损坏图片）。样式与当前显示设置一致。

- 数据来源，两种方式：
  - 上传文件：图片、JSON、点云可以一起选，也可以分几次选（会累加，同路径的去重）；整个文件夹可以选择或直接拖进窗口。图片和 JSON 也可以分开传：先传一部分，再用“追加到当前”补上另一部分，JSON 里写的路径、文件名或相机名会自动对上。
  - 压缩包：zip 不解压、按需读取；tar、tar.gz/tgz、tar.bz2、tar.xz 用内置库解压；7z、rar、tar.zst 调用系统自带的 bsdtar（Windows 10 1803 起自带 `tar.exe`，macOS 自带，Linux 装 `libarchive-tools`），没有时找 7-Zip。单个文件的 `.gz/.bz2/.xz`（如 `000.json.gz`）解压成原文件。压缩包里套压缩包会继续解开（最多 3 层）。加密或损坏的包会报出是哪个文件。
  - 打开本机目录：原地读取，不复制，适合几十 GB 的数据。支持目录、单个 json 和压缩包：zip 原地读取；其他格式解压到会话目录，原文件不动。桌面版可以点“浏览”选目录。
- 图片和 JSON 的对应：按 JSON 里写的图片路径找；找不到时退回按文件名、相机名匹配。找不到图片的帧、没被任何 JSON 引用的图片都会在侧栏提示。没有任何可用 JSON 时按图片浏览，`<序列>/<相机>/<帧>.jpg` 这样的目录会把同名图片归为同一帧的多个相机。
- 噪音过滤：只读取图片、JSON、点云和压缩包，其余文件（文档、日志、模型等）、隐藏文件，以及 `.git`、`__MACOSX`、`__pycache__`、`node_modules` 等目录会被忽略，上传前就在浏览器里过滤掉，侧栏显示忽略了多少。超过 64 MB 的 JSON 不当作单帧（多半是整个数据集的元数据表），跳过并说明原因。
- nuScenes：打开 nuScenes 数据根目录（含 `v1.0-*/` 和 `samples/`）即可，不需要转换。每个关键帧是一帧，所有相机、LIDAR_TOP 和 3D 框自动识别；每个传感器用自己的时间戳和位姿，投影与 nuscenes-devkit 的公式一致。元数据首次扫描一次（trainval 约 40 秒），索引缓存到数据目录，之后秒开。磁盘上缺文件的传感器不显示，并在侧栏列出。
- 环视：当前帧的所有相机按前左、前、前右 / 左、右 / 后排成网格，点击某个相机放大查看。
- 相机投影：点云按深度、强度或高度着色；显示 3D 框、折线、多边形和点，以及 2D 框 / 折线 / 多边形 / 点。3D 和 2D 标注可分别开关。支持滚轮缩放、拖拽平移、双击复位，←/→ 切换帧。鼠标悬停显示像素坐标和最近点的深度；悬停在标注上时显示类别、id 和属性（track_id、遮挡等）。
- 类别筛选：标注按大类归组并统一配色，分为四轮车、两轮车、行人、障碍物、车道线/路面和其他。每类显示 3D/2D 数量，可以单独隐藏。筛选同时作用于相机投影、BEV、3D 视图和下载的渲染图。类别名到大类的映射在 `config\categories.json` 里，可以按公司的标签体系增删（支持中英文、精确别名和关键词）。
- 时间同步 / 运动补偿（默认开）：相机、雷达各有时间戳时，用自车位姿把点云补偿到相机曝光时刻。点云带逐点时间时做运动去畸变，自动识别相对/绝对时间和 s/ms/us/ns 单位。状态栏下方会列出实际做了哪些补偿。
- 3D 点云：WebGL 显示点云（按高度或强度着色）、3D 框、轨迹和相机位置。左键旋转，右键或 Shift+左键平移，滚轮缩放。帧范围和坐标系与 BEV 共用。
- 外参检查：检查当前相机和雷达的外参约定是否读对（见下文）。批量检查会列出可疑帧，点击行跳转到该帧。
- BEV / 拼接：选择帧范围（如 `0-9,15`），统一到首帧自车系或世界系，生成俯视图或下载 PLY。

## 批量命令行（大数据量用这个）

```bash
python -m sensorlens.cli export data/batch.zip --out out/viz              # 投影图 + 环视 + BEV + 报告
python -m sensorlens.cli render data/batch.zip --out out/render --workers 8
python -m sensorlens.cli check  data/batch.zip --out out/report.csv --workers 8
python -m sensorlens.cli stitch data/batch.zip --frames 0-99 --target world --out out/map.ply --bev out/map.png
```

输入可以是目录、zip、单个 JSON 或 nuScenes 数据根目录。`--adapter` 指定格式（默认自动识别），`--frames`、`--cameras` 用来选择范围，`--no-sync` 关闭时间同步。多进程之间共享同一份帧索引，每个进程只读取自己负责的帧。

大数据集：JSON 数量达到 400 个时自动多进程建索引。`--cache out/index.json` 会把索引存到文件里，数据不变时下次直接读缓存（以文件大小和修改时间判断是否变化）。网页上传的会话会自动缓存索引。

## 标准 JSON 格式

```jsonc
{
  "extends": "calib.json",          // 可选：先合并公共标定文件，再用本文件覆盖
  "frame_id": "000123",
  "timestamp": 1700000000.1,
  "conventions": { "quat_order": "wxyz" },          // 见下表，文件级默认值
  "ego_pose": { "rotation": [w,x,y,z], "translation": [x,y,z] },   // 自车→世界
  "cameras": {                       // 列表或以名字为键的对象均可
    "CAM_FRONT": {
      "image": "images/CAM_FRONT/000123.jpg",       // 相对本 JSON 所在目录；找不到时按文件名匹配
      "width": 1920, "height": 1080,
      "intrinsic": { "fx": 1150, "fy": 1150, "cx": 960, "cy": 540 },   // 也可 3x3 / 扁平 9 / [fx,fy,cx,cy] / 3x4 投影矩阵 P
      "distortion": { "model": "opencv", "coeffs": [k1,k2,p1,p2,k3] }, // pinhole / opencv(最多8个) / fisheye(k1..k4)
      "extrinsic": { "rotation": [w,x,y,z], "translation": [x,y,z], "parent": "ego" }
    }
  },
  "lidars": {
    "LIDAR_TOP": { "path": "lidar/000123.bin", "dims": 4, "extrinsic": { "matrix": [[...4x4...]] } }
  },
  "annotations": [
    { "type": "box3d", "label": "car", "frame": "ego", "center": [x,y,z], "size": [l,w,h], "yaw": 0.1 },
    { "type": "polyline", "label": "lane", "frame": "ego", "points": [[x,y,z], ...] },   // 2D 点自动补 z=0
    { "type": "polygon",  "label": "crosswalk", "points": [...] },
    { "type": "points",   "label": "keypoint",  "points": [[x,y,z], ...] },
    // 2D 标注：像素坐标，camera 指定画在哪个相机上（只有一个相机时可省略）
    { "type": "box2d", "label": "pedestrian", "camera": "CAM_FRONT", "bbox": [x1,y1,x2,y2], "id": "7",
      "attributes": { "occlusion": 1 } },
    { "type": "polyline2d", "label": "lane", "camera": "CAM_FRONT", "points": [[u,v], ...] },
    { "type": "polygon2d" }, { "type": "points2d" }
  ]
}
```

最简写法（只有图片 + 2D 标注）：

```json
{ "image": "a.jpg", "annotations": [ { "type": "box2d", "label": "car", "bbox": [100, 200, 300, 400] } ] }
```

没有 `width` / `height` 时从图片读取尺寸。标注里 `attributes` 下的字段和其他标量字段（如 `occlusion`、`track_id`）都会在悬停时显示。

时间相关字段：帧和每个传感器都可以有 `timestamp`，传感器也可以有自己的 `ego_pose`（曝光时刻的位姿）。没有传感器位姿时，用同一序列（`sequence` 或 `scene` 字段，默认按 JSON 所在目录划分）各帧的位姿插值得到。雷达的 `point_time` 写逐点时间所在列，例如 `{"column": 4, "unit": "s", "mode": "relative"}`，也可以省略，此时自动从 PCD/PLY 的 `t` / `time` / `timestamp` 字段读取。

外参的写法：`matrix`（4x4、3x4 或扁平 16 个数），或 `rotation` + `translation`，或 `{"chain": [A, B, ...]}`（按书写顺序相乘 A·B·…，例如 KITTI 的 `R0_rect · Tr_velo_to_cam`）。近似正交的旋转（如保留 3 位小数）会自动正交化，偏差超过 1% 或行列式为负则报错。`rotation` 可以是四元数、3x3 矩阵、`[roll, pitch, yaw]`，也可以是 `{roll, pitch, yaw}` 或 `{w, x, y, z}`。`parent` 默认为 `ego`，也可以写另一个传感器名（例如相机外参是相对雷达给的）。标注的 `frame` 可以是 `ego`、`world` 或任意传感器名。

内参给 3x4 投影矩阵 `P = K [R | t]` 时（如 KITTI 的 `P2`），会分解出 K 和投影前的 `[R | t]`，不会丢掉 P 第 4 列里的基线平移。鱼眼相机默认最大半视场角 100°，更广的镜头用相机级 `max_fov_deg`（全视场角）指定。

3D 框：`size` 按 `box_size_order` 解读，朝向用 `yaw`（或 KITTI 的 `rotation_y`）或完整的 `rotation`。框所在坐标系由 `box_axes` 决定：`flu` 表示长沿 x、宽沿 y、高沿 z，yaw 绕 z；`rdf` 是 KITTI 相机系写法，长沿 x、高沿 -y、宽沿 z，yaw 绕 y。默认 `auto`：`frame` 是相机时按 rdf，否则按 flu。

点云格式：`.bin`（float32，列数由 `dims` 指定）、`.pcd`（ascii / binary / binary_compressed）、`.ply`、`.npy`、`.txt/.csv`。

### 约定（conventions）

可以写在文件级，也可以写在单个传感器的 `conventions` 或单个外参对象里覆盖（越具体越优先）。

| 键 | 默认 | 可选 | 说明 |
|---|---|---|---|
| `quat_order` | `wxyz` | `xyzw` | 四元数分量顺序 |
| `extrinsic_direction` | `sensor_to_parent` | `parent_to_sensor` | 外参把传感器系的点变到父系，还是反过来（别名 `sensor_to_ego` / `ego_to_sensor`） |
| `camera_axes` | `opencv` | `ros_body` | 相机外参描述的是光学系（x右 y下 z前），还是机体系（x前 y左 z上） |
| `matrix_order` | `row` | `col` | 扁平数组按行还是按列展开 |
| `euler_degrees` | `false` | `true` | 欧拉角单位 |
| `euler_order` | `zyx` | `xyz` | `[roll, pitch, yaw]` 的组合顺序：zyx 为 R = Rz(yaw)·Ry(pitch)·Rx(roll)（ROS rpy），xyz 为 Rx·Ry·Rz |
| `pose_direction` | `ego_to_world` | `world_to_ego` | ego_pose 的方向 |
| `box_size_order` | `lwh` | `wlh` / `hwl` / `whl` | 框尺寸顺序（nuScenes 为 wlh，KITTI 为 hwl） |
| `box_axes` | `auto` | `flu` / `rdf`（别名 kitti） | 见上文 3D 框 |
| `box_origin` | `center` | `bottom` | 框中心是几何中心还是底面中心 |
| `bbox_format` | `xyxy` | `xywh`（别名 coco）/ `cxcywh`（别名 yolo） | 2D 框格式 |
| `time_unit` | `auto` | `s` / `ms` / `us` / `ns` | 时间戳单位；auto 按数量级判断 |

自车系约定为 x 前、y 左、z 上。

## 任意格式的 JSON：自动识别

不同公司、不同数据集的 JSON 写法各不相同。不是标准格式、也没有匹配的适配器时，程序会抽样同一种格式的几个文件，按字段名和数值形状推断每样东西在哪里、该怎么读，然后照常可视化：

- 相机：图片路径看扩展名；内参看是否为上三角且焦距为正（3x3、扁平 9 个数、`fx/fy/cx/cy`、KITTI 的 3x4 投影矩阵 `P2` 都可以）；没写分辨率时从图片读取。
- 外参和位姿：4x4 / 3x4 矩阵，或旋转 + 平移（包括 `sensor2ego_rotation` + `sensor2ego_translation` 这种平铺写法）。方向从字段名读，例如 `cam2ego`、`lidar_to_cam`、`T_cam_ego`、`ego2global`；KITTI 的 `Tr_velo_to_cam` 会自动乘上 `R0_rect`。
- 标注：3 维中心 + 3 维尺寸是 3D 框，`[x,y,z,l,w,h,yaw]` 数组和与之并列的类别数组（`gt_boxes` + `gt_names`）也能识别；4 个数或 `xmin/ymin/xmax/ymax` 一类字段是 2D 框；N×2 / N×3 是折线、多边形或点。labelme 的 `shape_type` 和 COCO 的 images / annotations / categories 三张表也支持。
- 能从数值判断的歧义按全部样本统计：框尺寸顺序（最长边是长）、2D 框是 xyxy 还是 xywh、四元数是 wxyz 还是 xyzw、角度单位，以及框所在的坐标系（世界系、雷达系、KITTI 的相机系）。
- 只有相机标定、没有图片和标注的 JSON（如 `calib.json`）作为同目录各帧的公共标定；JSON 与图片同名时（`000123.json` ↔ `CAM_FRONT/000123.jpg`）自动配对。

推断依据会列在侧栏“JSON 格式自动识别”里，可以点“下载为适配器”保存成文件，检查或修改后放进 `adapters/` 固定下来。外参方向、四元数顺序、相机轴约定即使推断错了，打开时的自动标定校验也会用图像核对并修正；仍对不上的相机不画 3D 投影。字段名和数值都无法判断的格式会被跳过并说明原因，这时请写一个适配器。

## 接入新格式：适配器

不用改代码。在 `adapters\` 下放一个 JSON，把源字段映射到标准字段，页面和命令行都能选到它。参考 `adapters\vendor_demo.json`：

- `detect`：自动识别时，这些路径都存在才使用此适配器。
- `frames`：一个 JSON 里有多帧时，指向帧列表（如 `"data[*]"`）。
- `fields` / `cameras` / `lidars` / `annotations`：`items` 指向元素列表，`fields` 是“标准字段 → 源路径”。
- 路径语法：`a.b[0].c`；`[*]` 遍历数组；`.*` 遍历对象的值；`$key` 取当前元素的键；`$root.x` 从文档根取值，`$frame.x` 从当前帧取值；路径里的 `[$key]` 换成当前元素的下标（并列数组 `names[$key]`）；`=opencv` 表示常量；写成数组时取第一个存在的值；`{"$array": [路径, ...]}` 把几个值组成数组。
- `where`：`{路径: 正则}`，只取匹配的元素（例如同一个传感器列表里只取相机）；`group`：先遍历分组（例如每个相机），`items` 在组内取值，`$groupkey` 是组名。

## 外参检查原理

外参出错最常见的原因是约定读反，而不是数值偏差。检查逻辑会对相机和雷达分别尝试“外参方向、四元数顺序、相机轴约定”的所有组合，每个组合打一个分。打分方法按数据自动选择：

1. 有点云：点云投到图像后，图像灰度和雷达反射强度之间的互信息。车道线这类在两种传感器里都更亮的区域，在约定正确时才会对齐。点云没有强度时，改用“深度跳变是否落在图像边缘上”。
2. 没有点云，但有 3D 框和同一相机的 2D 框：3D 框投影的外接矩形与 2D 框的 IoU（按 id 配对，没有 id 时按类别）。
3. 只有 3D 标注：投影线与图像边缘的重合度，置信度较低。
4. 只有 2D 标注或没有内外参：无法检查。

再乘以物理先验：雷达系下大部分点应在雷达下方；相机图像的“下”方向应朝向地面，光轴不应竖直。

得分最高的组合若明显优于当前约定，就判为“疑似有误”并给出具体改法。这个检查针对约定错误，不能用来评估几个像素级别的标定精度。

## 目录

```
sensorlens/  geometry 变换与坐标系图 | camera 相机模型 | pointcloud 点云读写
             schema 标准格式解析 | adapter 字段映射 | automap 未知格式自动识别 | dataset 目录/zip 索引 | archives 压缩包解压
             render 投影/BEV/拼接/时间同步 | motion 位姿插值与去畸变 | categories 类别归组
             calibcheck 外参检查 | autocalib 自动标定校验 | integrity 图片完整性 | plan 视图选择
             export 导出 | server Web 服务 | cli 批量 | demo 演示数据
web/         前端（原生 JS + Canvas + WebGL，无需构建）
desktop/     桌面外壳（Tauri）
adapters/    格式适配器
config/      类别配置 categories.json
docs/        图标等文档资源
samples/     演示数据
tests/       单元测试
data/        上传会话（运行时生成）
.cache/      临时文件与缓存（运行时生成，可随时删除）
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
