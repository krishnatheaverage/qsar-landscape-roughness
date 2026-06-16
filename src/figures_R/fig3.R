#!/usr/bin/env Rscript
# fig3.R -- color-independent ggplot2 version of manuscript Figure 3 (robustness).
#
# Reproduces src/robustness_analyze.py's matplotlib figure:
#   3 panels (local_var, nbr_disp, nbr_disp|AD) of median Spearman rho vs RF error
#   plotted against neighbourhood size k, for 3 series
#   (ECFP4/Tanimoto, ECFP6/Tanimoto, RDKit-desc/Euclidean).
#
# The plotted values ARE the columns of results/robustness_summary.csv, which the
# Python script writes from the very same DataFrame it plots. We read that CSV
# directly -- no statistics are re-derived in R.
#
# Output: paper/figures/robustness_figure.png  (filename referenced by the manuscript).

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

# ---- resolve repo root relative to this script, so it runs from a fresh clone ----
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_path <- if (length(file_arg) == 1) normalizePath(file_arg) else NA
root <- if (!is.na(script_path)) {
  normalizePath(file.path(dirname(script_path), "..", ".."))
} else {
  normalizePath(getwd())
}

csv_path <- file.path(root, "results", "robustness_summary.csv")
png_path <- file.path(root, "paper", "figures", "robustness_figure.png")
stopifnot(file.exists(csv_path))

# ---- load the precomputed summary ----
R <- read_csv(csv_path, show_col_types = FALSE)

# series factor with the same display names + order as the matplotlib NAME dict
metric_levels <- c("ecfp4", "ecfp6", "desc")
metric_names  <- c(ecfp4 = "ECFP4 / Tanimoto",
                   ecfp6 = "ECFP6 / Tanimoto",
                   desc  = "RDKit desc / Euclidean")

# long form: one row per (metric, k, panel)
long <- R %>%
  rename(`local_var` = local_var,
         `nbr_disp` = nbr_disp,
         `nbr_disp | AD` = nbr_disp_partial) %>%
  pivot_longer(c(`local_var`, `nbr_disp`, `nbr_disp | AD`),
               names_to = "panel", values_to = "rho") %>%
  mutate(
    metric = factor(metric, levels = metric_levels),
    series = factor(metric_names[as.character(metric)],
                    levels = unname(metric_names[metric_levels])),
    panel  = factor(panel, levels = c("local_var", "nbr_disp", "nbr_disp | AD"))
  )

k_breaks <- sort(unique(R$k))  # 5 10 15 20 30

# grayscale-safe + Okabe-Ito encodings (one entry per series, in display order)
series_levels <- levels(long$series)
lty_vals   <- c("solid", "22", "dotted")            # distinct linetypes
shape_vals <- c(16, 17, 15)                          # circle, triangle, square
oi_cols    <- c("#000000", "#0072B2", "#D55E00")     # Okabe-Ito (ADDITION only)
names(lty_vals) <- names(shape_vals) <- names(oi_cols) <- series_levels

panel_titles <- c(
  "local_var"      = "local_var  (landscape roughness, uses y)",
  "nbr_disp"       = "nbr_disp  (y-free)",
  "nbr_disp | AD"  = "nbr_disp | AD  (partial, controls NN-dist + density)"
)

base_theme <- theme_bw(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title  = element_text(face = "bold", size = rel(1.0)),
        legend.title = element_text(size = rel(0.85)),
        legend.position = "bottom",
        legend.key.width = unit(1.6, "lines"))

# shared y range across panels (matplotlib used sharey=True)
y_rng <- range(c(0, long$rho), na.rm = TRUE)

make_panel <- function(panel_key, show_y = FALSE) {
  d <- filter(long, panel == panel_key)
  p <- ggplot(d, aes(x = k, y = rho,
                     linetype = series, shape = series, colour = series)) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.6, fill = "white", stroke = 0.8) +
    scale_x_continuous(breaks = k_breaks) +
    scale_linetype_manual(values = lty_vals, name = "descriptor / metric") +
    scale_shape_manual(values = shape_vals, name = "descriptor / metric") +
    scale_colour_manual(values = oi_cols, name = "descriptor / metric") +
    coord_cartesian(ylim = y_rng) +
    labs(title = panel_titles[[panel_key]], x = "neighbourhood size k", y = NULL) +
    base_theme
  if (show_y) p <- p + labs(y = expression("median Spearman " * rho * " vs RF error"))
  p
}

p1 <- make_panel("local_var", show_y = TRUE)
p2 <- make_panel("nbr_disp")
p3 <- make_panel("nbr_disp | AD")

fig <- (p1 | p2 | p3) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Robustness: the roughness-error relationship is stable across k and across distance metric",
    tag_levels = "A",
    theme = theme(plot.title = element_text(face = "bold", size = 15, hjust = 0.5))
  ) &
  theme(legend.position = "bottom")

W <- 15; H <- 5.2
ggsave(png_path, plot = fig, width = W, height = H, dpi = 300)
cat("saved ->", png_path, "\n")
cat(sprintf("series order: %s\n", paste(series_levels, collapse = " | ")))
