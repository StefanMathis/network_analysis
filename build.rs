fn main() {
    // If building for docs.rs, DO NOT create the README files from the template
    if let Ok(env) = std::env::var("DOCS_RS") {
        if &env == "1" {
            return ();
        }
    }

    let mut readme = std::fs::read_to_string("README.template.md").unwrap();
    readme = readme.replace(
        "{{VERSION}}",
        std::env::var("CARGO_PKG_VERSION")
            .expect("version is available in build.rs")
            .as_str(),
    );

    // Generate README_local.md using local images
    let mut local = readme.replace("{{example.svg}}", "docs/example.svg");
    local = local.replace("{{graph_terminology.svg}}", "docs/graph_terminology.svg");
    std::fs::write("README_local.md", local).unwrap();

    // Generate README,md using online hosted images
    let mut docsrs = readme.replace(
        "{{example.svg}}",
        "https://raw.githubusercontent.com/StefanMathis/network_analysis/main/docs/example.svg",
    );
    docsrs = docsrs.replace("{{graph_terminology.svg}}", "https://raw.githubusercontent.com/StefanMathis/network_analysis/main/docs//graph_terminology.svg");
    std::fs::write("README.md", docsrs).unwrap();
}
