//! 桌面外壳：管理 Python 后端进程、数据目录配置和在线更新。
//!
//! 窗口先加载内置的启动页（ui/index.html）。启动页确认数据目录后调用 `start`，
//! 这里在 127.0.0.1 的随机端口上拉起后端，再把窗口导航到后端页面，URL 带一次性令牌。

use std::fs::{self, File};
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, RunEvent, Url, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use tauri_plugin_updater::UpdaterExt;

const RELEASES_URL: &str = "https://github.com/Apageoflove/ad-viz/releases/latest";

#[derive(Default, Serialize, Deserialize, Clone)]
struct Config {
    data_dir: Option<PathBuf>,
}

#[derive(Default)]
struct Backend {
    child: Mutex<Option<Child>>,
    url: Mutex<Option<String>>,
    save_path: Mutex<Option<PathBuf>>,
}

/// 便携版：exe 同目录下有 `.portable` 文件。配置、WebView 缓存都放在 exe 旁边，不写系统目录。
fn portable_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    dir.join(".portable").exists().then(|| dir.to_path_buf())
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(d) = portable_dir() {
        return Ok(d.join("ad-viz-config.json"));
    }
    let dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    Ok(dir.join("config.json"))
}

fn load_config(app: &AppHandle) -> Config {
    config_path(app)
        .ok()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_config(app: &AppHandle, cfg: &Config) -> Result<(), String> {
    let p = config_path(app)?;
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let tmp = p.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(cfg).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &p).map_err(|e| e.to_string())
}

fn backend_exe(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("ADVIZ_BACKEND") {
        return Ok(PathBuf::from(p));
    }
    let name = if cfg!(windows) { "adviz-backend.exe" } else { "adviz-backend" };
    let mut candidates = Vec::new();
    if let Ok(res) = app.path().resource_dir() {
        candidates.push(res.join("backend").join(name));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join("backend").join(name));
    }
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| "找不到后端程序 backend/adviz-backend，安装可能不完整，请重新安装".to_string())
}

fn log_tail(path: &Path) -> String {
    let mut s = String::new();
    if let Ok(mut f) = File::open(path) {
        let _ = f.read_to_string(&mut s);
    }
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(15)..].join("\n")
}

fn stop_backend(app: &AppHandle) {
    let st = app.state::<Backend>();
    if let Some(mut c) = st.child.lock().unwrap().take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    *st.url.lock().unwrap() = None;
}

/// 启动后端并等待端口可连接，返回带令牌的页面地址。
fn start_backend(app: &AppHandle) -> Result<String, String> {
    let st = app.state::<Backend>();
    if let Some(u) = st.url.lock().unwrap().clone() {
        let alive = st.child.lock().unwrap().as_mut().map(|c| matches!(c.try_wait(), Ok(None))).unwrap_or(false);
        if alive {
            return Ok(u);
        }
    }
    stop_backend(app);
    let data = load_config(app).data_dir.ok_or("还没有选择数据目录")?;
    fs::create_dir_all(data.join("logs")).map_err(|e| format!("无法创建数据目录 {}: {e}", data.display()))?;
    let exe = backend_exe(app)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(m) = fs::metadata(&exe) {
            if m.permissions().mode() & 0o111 == 0 {
                let _ = fs::set_permissions(&exe, fs::Permissions::from_mode(0o755));
            }
        }
    }
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| format!("无法分配本机端口: {e}"))?
        .port();
    let token = uuid::Uuid::new_v4().simple().to_string();
    let log_path = data.join("logs").join("backend.log");
    let log = File::create(&log_path).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(&exe);
    cmd.arg("serve")
        .arg("--port")
        .arg(port.to_string())
        .arg("--data")
        .arg(&data)
        .arg("--watch-stdin")
        .env("ADVIZ_TOKEN", &token)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let child = cmd.spawn().map_err(|e| format!("启动后端失败 {}: {e}", exe.display()))?;
    *st.child.lock().unwrap() = Some(child);

    let addr = format!("127.0.0.1:{port}").parse().unwrap();
    let t0 = Instant::now();
    loop {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok() {
            break;
        }
        if let Some(c) = st.child.lock().unwrap().as_mut() {
            if let Ok(Some(code)) = c.try_wait() {
                return Err(format!("后端退出（{code}）。日志 {}:\n{}", log_path.display(), log_tail(&log_path)));
            }
        }
        if t0.elapsed() > Duration::from_secs(120) {
            stop_backend(app);
            return Err(format!("后端 120 秒内没有就绪。日志 {}:\n{}", log_path.display(), log_tail(&log_path)));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let url = format!("http://127.0.0.1:{port}/?token={token}");
    *st.url.lock().unwrap() = Some(url.clone());
    Ok(url)
}

fn navigate(app: &AppHandle, url: &str) -> Result<(), String> {
    let win = app.get_webview_window("main").ok_or("主窗口不存在")?;
    win.navigate(Url::parse(url).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn pick_dir(app: &AppHandle) -> Result<Option<PathBuf>, String> {
    let picked = app.dialog().file().set_title("选择 ad-viz 数据目录（会话、缓存、日志）").blocking_pick_folder();
    match picked {
        Some(p) => Ok(Some(p.into_path().map_err(|e| e.to_string())?)),
        None => Ok(None),
    }
}

#[tauri::command]
fn app_info(app: AppHandle) -> Value {
    let cfg = load_config(&app);
    json!({
        "version": app.package_info().version.to_string(),
        "data_dir": cfg.data_dir,
        "default_data_dir": default_data_dir(&app),
        "portable": portable_dir().is_some(),
        "os": std::env::consts::OS,
    })
}

fn default_data_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Some(d) = portable_dir() {
        return Some(d.join("ad-viz-data"));
    }
    app.path().app_local_data_dir().ok()
}

#[tauri::command]
async fn choose_data_dir(app: AppHandle, use_default: Option<bool>) -> Result<Option<String>, String> {
    let dir = if use_default.unwrap_or(false) {
        default_data_dir(&app)
    } else {
        let a = app.clone();
        tauri::async_runtime::spawn_blocking(move || pick_dir(&a)).await.map_err(|e| e.to_string())??
    };
    let Some(dir) = dir else { return Ok(None) };
    fs::create_dir_all(&dir).map_err(|e| format!("无法创建 {}: {e}", dir.display()))?;
    save_config(&app, &Config { data_dir: Some(dir.clone()) })?;
    Ok(Some(dir.display().to_string()))
}

#[tauri::command]
async fn start(app: AppHandle) -> Result<(), String> {
    let a = app.clone();
    let url = tauri::async_runtime::spawn_blocking(move || start_backend(&a)).await.map_err(|e| e.to_string())??;
    navigate(&app, &url)
}

/// 切换数据目录：旧目录原样保留（可以随时切回去），后端用新目录重启。
#[tauri::command]
async fn change_data_dir(app: AppHandle) -> Result<Option<String>, String> {
    let a = app.clone();
    let Some(dir) = tauri::async_runtime::spawn_blocking(move || pick_dir(&a)).await.map_err(|e| e.to_string())?? else {
        return Ok(None);
    };
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    save_config(&app, &Config { data_dir: Some(dir.clone()) })?;
    let a = app.clone();
    let url = tauri::async_runtime::spawn_blocking(move || {
        stop_backend(&a);
        start_backend(&a)
    })
    .await
    .map_err(|e| e.to_string())??;
    let a = app.clone();
    // 先把结果返回给页面，再导航，否则页面拿不到返回值
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        let _ = navigate(&a, &url);
    });
    Ok(Some(dir.display().to_string()))
}

#[tauri::command]
async fn check_update(app: AppHandle) -> Result<Value, String> {
    let up = app.updater().map_err(|e| e.to_string())?.check().await.map_err(|e| e.to_string())?;
    Ok(match up {
        Some(u) => json!({
            "available": true,
            "version": u.version,
            "notes": u.body,
            "portable": portable_dir().is_some(),
        }),
        None => json!({ "available": false }),
    })
}

/// 下载并安装更新。数据目录和配置不在安装目录里，覆盖安装不会动它们。
#[tauri::command]
async fn install_update(app: AppHandle) -> Result<(), String> {
    if portable_dir().is_some() {
        // 便携版没有安装器可以调用，打开下载页让用户替换文件夹
        app.opener().open_url(RELEASES_URL, None::<&str>).map_err(|e| e.to_string())?;
        return Ok(());
    }
    let u = app
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?
        .ok_or("已是最新版本")?;
    let a = app.clone();
    let mut done: u64 = 0;
    let bytes = u
        .download(
            move |chunk, total| {
                done += chunk as u64;
                let _ = a.emit("update-progress", json!({ "done": done, "total": total }));
            },
            || {},
        )
        .await
        .map_err(|e| e.to_string())?;
    // Windows 上安装器要覆盖 backend 目录，必须先结束后端进程
    stop_backend(&app);
    u.install(bytes).map_err(|e| e.to_string())?;
    app.restart();
}

#[tauri::command]
async fn pick_save_path(app: AppHandle, name: String) -> Result<Option<String>, String> {
    let a = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || a.dialog().file().set_file_name(&name).blocking_save_file())
        .await
        .map_err(|e| e.to_string())?;
    let path = match picked {
        Some(p) => p.into_path().map_err(|e| e.to_string())?,
        None => return Ok(None),
    };
    *app.state::<Backend>().save_path.lock().unwrap() = Some(path.clone());
    Ok(Some(path.display().to_string()))
}

/// 写入 pick_save_path 选定的文件。数据以原始字节传入，避免大文件走 JSON。
#[tauri::command]
fn save_file(app: AppHandle, request: tauri::ipc::Request<'_>) -> Result<(), String> {
    let path = app.state::<Backend>().save_path.lock().unwrap().take().ok_or("没有选择保存位置")?;
    let tauri::ipc::InvokeBody::Raw(data) = request.body() else {
        return Err("数据格式错误".into());
    };
    fs::write(&path, data).map_err(|e| format!("写入 {} 失败: {e}", path.display()))
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Backend::default())
        .invoke_handler(tauri::generate_handler![
            app_info,
            start,
            choose_data_dir,
            change_data_dir,
            check_update,
            install_update,
            pick_save_path,
            save_file
        ])
        .setup(|app| {
            let mut b = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("ad-viz")
                .inner_size(1440.0, 900.0)
                .min_inner_size(900.0, 600.0)
                // 让网页自己处理拖入文件（上传区），而不是被外壳拦截
                .disable_drag_drop_handler();
            if let Some(d) = portable_dir() {
                b = b.data_directory(d.join("webview-data"));
            }
            b.build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building ad-viz");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            stop_backend(handle);
        }
    });
}
