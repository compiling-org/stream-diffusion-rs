//! Stream Diffusion RS - Main Application
//!
//! This is the main entry point for the Stream Diffusion RS application.
//! It starts the web server directly with automatic browser launch.

use std::process::Command;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Stream Diffusion RS - Advanced Multimodal AI Toolkit");
    println!("==========================================");
    println!("🌐 Starting web server...");
    println!("💡 Press Ctrl+C to stop the server");
    println!("🔗 Web interface will open automatically at http://127.0.0.1:3000");


    // Launch browser after a short delay to let server start
    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        if let Err(e) = launch_browser() {
            eprintln!("⚠️  Warning: Could not launch browser automatically: {}", e);
        }
    });

    // Start the web server
    match stream_diffusion_rs::web::start_default_server().await {
        Ok(_) => {
            println!("✅ Server stopped gracefully");
        }
        Err(e) => {
            eprintln!("❌ Server error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

/// Launch browser automatically
fn launch_browser() -> Result<(), Box<dyn std::error::Error>> {
    let url = "http://127.0.0.1:3000";

    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(&["/C", "start", url])
            .spawn()?;
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(url)
            .spawn()?;
    }

    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(url)
            .spawn()?;
    }

    Ok(())
}