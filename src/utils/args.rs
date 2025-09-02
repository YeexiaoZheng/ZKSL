use clap::Parser;

/// A simple CLI program
#[derive(Parser)]
#[command(version = "1.0", about = "CLI arguments")]
pub struct Cli {
    // /// An optional name argument
    // #[arg(short, long, default_value = "world")]
    // name: String,
    /// A required number argument
    #[arg(long)]
    pub parallel: bool,

    /// A required number argument
    #[arg(long)]
    pub no_prove: bool,

    /// An optional flag argument
    #[arg(short, long, default_value = "1")]
    pub batch_size: usize,
}
