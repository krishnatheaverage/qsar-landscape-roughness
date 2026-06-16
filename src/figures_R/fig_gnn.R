#!/usr/bin/env Rscript
# Print-safe hatched ggplot2 rebuild of the GNN fixed-vs-tuned figure.

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
this_file <- sub("^--file=", "", args[grep("^--file=", args)])
if (length(this_file) == 0) this_file <- "src/figures_R/fig_gnn.R"
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))
RESULTS_DIR <- file.path(ROOT, "results")
FIG_DIR     <- file.path(ROOT, "paper", "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
png_path <- file.path(FIG_DIR, "gnn_tuned_figure.png")

base_theme <- theme_bw(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = rel(0.92)),
        plot.title.position = "plot")

make_hatch <- function(xmin, xmax, ymin, ymax, pattern, sx = 0.055, sy = 0.020) {
  out <- list()
  if (pattern %in% c("vert", "cross")) {
    xs <- seq(xmin + sx, xmax - 1e-9, by = sx)
    if (length(xs)) out$v <- data.frame(x = xs, xend = xs, y = ymin, yend = ymax)
  }
  if (pattern %in% c("horiz", "cross")) {
    ys <- seq(ymin + sy, ymax - 1e-9, by = sy)
    if (length(ys)) out$h <- data.frame(x = xmin, xend = xmax, y = ys, yend = ys)
  }
  if (length(out)) do.call(rbind, out) else NULL
}

pa <- read_csv(file.path(RESULTS_DIR, "gnn_fixed_vs_tuned.csv"), show_col_types = FALSE)
n  <- nrow(pa)
nfix <- sum(pa$mae_fixed > pa$mae_rf); ntun <- sum(pa$mae_tuned > pa$mae_rf)
lab <- if (nfix == n && ntun == n) {
  sprintf("both regimes: GNN MAE > RF on %d/%d", n, n)
} else {
  sprintf("GNN MAE > RF: fixed %d/%d, tuned %d/%d", nfix, n, ntun, n)
}
lim <- c(0.2, 1.25)

pA <- ggplot(pa, aes(mae_fixed, mae_tuned)) +
  geom_segment(x = lim[1], y = lim[1], xend = lim[2], yend = lim[2],
               linetype = "dashed", colour = "grey30", linewidth = 0.4) +
  geom_point(shape = 21, fill = "grey15", colour = "white", size = 2.6, stroke = 0.4) +
  annotate("text", x = lim[1] + 0.02, y = lim[2] - 0.02, label = lab,
           hjust = 0, vjust = 1, size = 3.0, colour = "grey25") +
  scale_x_continuous(limits = lim, expand = c(0, 0)) +
  scale_y_continuous(limits = lim, expand = c(0, 0)) +
  labs(title = "Tuned vs fixed-60 GNN MAE\n(on/above diagonal: tuning gave no systematic gain)",
       x = "GNN test MAE, fixed 60 epochs",
       y = "GNN test MAE, tuned (early stopping)") +
  base_theme

pb <- read_csv(file.path(RESULTS_DIR, "gnn_quartile_gap.csv"), show_col_types = FALSE)
pb$regime <- factor(pb$regime, levels = c("fixed-60", "tuned"))
bw <- 0.38; off <- 0.20
bars <- pb %>% mutate(
  center  = rough_q + ifelse(regime == "fixed-60", -off, off),
  xmin    = center - bw / 2, xmax = center + bw / 2,
  ymin    = 0, ymax = mean_gap,
  pattern = ifelse(regime == "fixed-60", "vert", "cross"))

segs <- do.call(rbind, lapply(seq_len(nrow(bars)), function(i)
  make_hatch(bars$xmin[i], bars$xmax[i], bars$ymin[i], bars$ymax[i], bars$pattern[i])))

leg <- data.frame(regime = c("fixed-60", "tuned"),
                  xmin = 3.18, xmax = 3.44,
                  ymin = c(0.262, 0.232), ymax = c(0.282, 0.252),
                  pattern = c("vert", "cross"))
legsegs <- do.call(rbind, lapply(seq_len(nrow(leg)), function(i)
  make_hatch(leg$xmin[i], leg$xmax[i], leg$ymin[i], leg$ymax[i], leg$pattern[i],
             sx = 0.05, sy = 0.013)))

pB <- ggplot() +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
  geom_rect(data = bars, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = "white", colour = "black", linewidth = 0.4) +
  geom_segment(data = segs, aes(x = x, y = y, xend = xend, yend = yend),
               colour = "grey25", linewidth = 0.22) +
  geom_rect(data = bars, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = NA, colour = "black", linewidth = 0.4) +
  geom_errorbar(data = bars, aes(x = center, ymin = mean_gap - sem, ymax = mean_gap + sem),
                width = 0.12, colour = "black", linewidth = 0.4) +
  geom_rect(data = leg, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = "white", colour = "black", linewidth = 0.4) +
  geom_segment(data = legsegs, aes(x = x, y = y, xend = xend, yend = yend),
               colour = "grey25", linewidth = 0.22) +
  geom_rect(data = leg, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = NA, colour = "black", linewidth = 0.4) +
  annotate("text", x = 3.50, y = c(0.272, 0.242),
           label = c("fixed-60", "tuned"), hjust = 0, size = 3.1) +
  scale_x_continuous(breaks = 1:4, labels = c("Q1\nsmooth", "Q2", "Q3", "Q4\nrough"),
                     limits = c(0.45, 4.55), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 0.30), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Gap still shrinks from smooth to rough after tuning",
       x = NULL, y = "mean (GNN error - RF error)") +
  base_theme

fig <- (pA | pB) +
  plot_layout(widths = c(1, 1)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 13))

W <- 10; H <- 5.2
ggsave(png_path, fig, width = W, height = H, dpi = 300)
cat("saved:", png_path, "\n")
cat("Panel B values used:\n"); print(pb)
