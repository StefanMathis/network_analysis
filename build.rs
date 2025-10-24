use std::{env, fs};

/**
This build script makes sure that images are sourced locally when building the
docs locally and from Github when building the images on docs.rs
 */
fn main() {
    let readme = fs::read_to_string("README.md").unwrap();

    let processed = if env::var("DOCS_RS").is_ok() {
        readme.replace(
            "](docs/",
            "](https://raw.githubusercontent.com/StefanMathis/network_analysis/main/docs/",
        )
    } else {
        readme
    };

    fs::write("target/README_FOR_DOCS.md", processed).unwrap();
    println!("cargo:rustc-env=README_PATH=target/README_FOR_DOCS.md");
}
