#!/usr/bin/env Rscript
# ggplot2 port of Figure 4: grouped grayscale bars for cross-domain validation.

suppressMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg)) normalizePath(dirname(file_arg)) else getwd()
root <- normalizePath(file.path(script_dir, "..", ".."))
results_dir <- file.path(root, "results")
fig_dir <- file.path(root, "paper", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
png_path <- file.path(fig_dir, "figure4_crossdomain.png")

df <- read_csv(file.path(results_dir, "figure4_plot_values.csv"),
               show_col_types = FALSE)

group_levels <- df %>%
  distinct(group, group_order) %>%
  arrange(group_order) %>%
  pull(group)
group_labels_wrapped <- c(
  "Bioactivity (30-target median)" = "Bioactivity\n(30-target median)",
  "ESOL (solubility)"              = "ESOL\n(solubility)",
  "Lipophilicity (logD)"           = "Lipophilicity\n(logD)"
)
df <- df %>%
  mutate(
    group  = factor(group, levels = group_levels),
    series = factor(series, levels = c("zero-order", "partial|AD"),
                    labels = c("zero-order ρ", "partial ρ | AD"))
  )

base_theme <- theme_bw(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = rel(1.0)),
    legend.title     = element_text(size = rel(0.85)),
    legend.key.size  = unit(0.9, "lines")
  )

fill_vals <- c("zero-order ρ" = "grey20", "partial ρ | AD" = "grey68")
y_lim <- c(0, 0.75)
y_lab <- "Spearman ρ vs per-compound error"

make_panel <- function(d, title, show_ylab) {
  ggplot(d, aes(x = group, y = value, fill = series)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.72,
             colour = "black", linewidth = 0.3) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    scale_fill_manual(values = fill_vals, name = NULL) +
    scale_x_discrete(labels = group_labels_wrapped) +
    scale_y_continuous(limits = y_lim, expand = expansion(mult = c(0, 0.02))) +
    labs(title = title, x = NULL,
         y = if (show_ylab) y_lab else NULL) +
    base_theme +
    theme(
      legend.background   = element_blank(),
      legend.key          = element_blank(),
      axis.text.x         = element_text(size = rel(0.82))
    )
}

pA <- make_panel(
  filter(df, panel == "A"),
  "Landscape roughness (Dirichlet)",
  show_ylab = TRUE
)
pB <- make_panel(
  filter(df, panel == "B"),
  "Activity-free roughness (nbr dispersion)",
  show_ylab = FALSE
)

fig <- (pA | pB) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "External validation: the roughness–error relationship replicates outside bioactivity",
    tag_levels = "A",
    theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
  ) &
  theme(legend.position = "bottom",
        plot.tag = element_text(face = "bold", size = 14))

ggsave(png_path, fig, width = 10, height = 5.3, dpi = 300)
cat("wrote", png_path, "\n")
