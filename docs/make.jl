using Documenter, BayesJL

repository_has_commit = success(pipeline(
    `git -C $(dirname(@__DIR__)) rev-parse HEAD`,
    stdout = devnull,
    stderr = devnull,
))
html_format = repository_has_commit ?
    Documenter.HTML(repolink = "git@github.com:CVerhoosel/BayesJL.git", edit_link = "main") :
    Documenter.HTML(repolink = nothing, edit_link = nothing)

makedocs(
    sitename = "BayesJL Documentation",
    modules = [BayesJL],
    format = html_format,
    remotes = repository_has_commit ? Dict() : nothing,
    pages = [
        "Home" => "index.md",
    ],
)