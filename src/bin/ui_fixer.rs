//! Command-line interface for the UI Fixer

use stream_diffusion_rs::ui_fixer;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 Stream Diffusion RS - UI Fixer");
    println!("=================================");
    
    // Get current directory as project root
    let project_root = std::env::current_dir()?.to_string_lossy().to_string();
    println!("Fixing UI implementations in project at: {}", project_root);
    
    // Run fixes
    match ui_fixer::run_ui_fixes(&project_root) {
        Ok(results) => {
            println!("\n🔧 UI Fix Results:");
            println!("=================");
            
            let successful = results.iter().filter(|r| r.success).count();
            let failed = results.len() - successful;
            
            println!("Total Components: {}", results.len());
            println!("Successfully Fixed: {}", successful);
            println!("Failed Fixes: {}", failed);
            
            println!("\nDetailed Results:");
            for result in &results {
                let status = if result.success { "✅" } else { "❌" };
                println!("{} {}", status, result.component);
                println!("  {}", result.message);
                
                if !result.files_modified.is_empty() {
                    println!("  Modified files:");
                    for file in &result.files_modified {
                        println!("    - {}", file);
                    }
                }
                println!();
            }
            
            // Save fix report
            let report_path = Path::new("ui_fix_report.txt");
            match ui_fixer::generate_fix_report(&project_root, &results, report_path) {
                Ok(_) => println!("📝 Fix report saved to: {}", report_path.display()),
                Err(e) => eprintln!("⚠️  Failed to save report: {}", e),
            }
            
            if failed > 0 {
                println!("\n⚠️  Some fixes failed. Please check the report for details.");
                std::process::exit(1);
            } else {
                println!("\n🎉 All UI fixes applied successfully!");
            }
        }
        Err(e) => {
            eprintln!("❌ Fix process failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}