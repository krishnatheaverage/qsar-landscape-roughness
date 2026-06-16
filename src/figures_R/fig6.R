#!/usr/bin/env Rscript
# Grayscale-safe ggplot2 rebuild of manuscript Figure 6 (conformal coverage panels).

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
this_file <- sub("^--file=", "", args[grep("^--file=", args)])
if (length(this_file) == 0) this_file <- "src/figures_R/fig6.R"
ROOT <- normalizePath(file.path(dirname(this_file), "..", ".."))
RESULTS_DIR <- file.path(ROOT, "results")
FIG_DIR     <- file.path(ROOT, "paper", "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
png_path <- file.path(FIG_DIR, "figure6_conformal.png")

okabe_ito <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
base_theme <- theme_bw(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title       = element_text(face = "bold", size = rel(1.0)),
        legend.title     = element_text(size = rel(0.85)))

curve <- read_csv(file.path(RESULTS_DIR, "conformal_curve.csv"),
                  show_col_types = FALSE)

A_order  <- c("standard", "variance", "AD", "roughness")
A_labels <- c(standard   = "unconditional",
              variance    = "tree variance",
              AD          = "applicability domain",
              roughness   = "roughness")

curveA <- curve %>%
  filter(method %in% A_order) %>%
  mutate(series = factor(A_labels[method], levels = unname(A_labels[A_order])),
         qbin   = as.integer(qbin))

q_labels <- c("Q1\n(smooth)", "Q2", "Q3", "Q4", "Q5\n(rough)")

A_linetypes <- c("solid", "22", "dotted", "dotdash")
A_shapes    <- c(16, 17, 15, 4)
A_colors    <- okabe_ito[c(1, 4, 6, 3)]
names(A_linetypes) <- names(A_shapes) <- names(A_colors) <- unname(A_labels[A_order])

pA <- ggplot(curveA, aes(x = qbin, y = cov,
                         linetype = series, shape = series, colour = series)) +
  geom_hline(yintercept = 0.90, linetype = "dashed", colour = "black", linewidth = 0.5) +
  annotate("text", x = 4, y = 0.905, label = "nominal 90%",
           hjust = 1, vjust = 0, size = 3.1, colour = "black") +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.8, stroke = 0.9, fill = "white") +
  scale_x_continuous(breaks = 0:4, labels = q_labels,
                     expand = expansion(mult = c(0.04, 0.04))) +
  scale_linetype_manual(values = A_linetypes, name = "conditioning variable") +
  scale_shape_manual(values = A_shapes,       name = "conditioning variable") +
  scale_colour_manual(values = A_colors,      name = "conditioning variable") +
  labs(title = "Coverage across the roughness range",
       x = "local roughness quantile",
       y = "90% interval coverage") +
  base_theme +
  theme(legend.position = c(0.02, 0.02),
        legend.justification = c(0, 0),
        legend.background = element_rect(fill = scales::alpha("white", 0.7), colour = NA),
        legend.key.width = unit(1.4, "lines"),
        legend.text = element_text(size = rel(0.8)))

res <- read_csv(file.path(RESULTS_DIR, "conformal_results.csv"),
                show_col_types = FALSE)

B_map <- c("standard (unconditional)"        = "none",
           "Mondrian: tree variance"          = "var.",
           "Mondrian: applicability domain"   = "AD",
           "Mondrian: roughness"              = "rough.",
           "Mondrian: variance + roughness"   = "var.+\nrough.")
B_order <- unname(B_map)

barB <- res %>%
  filter(method %in% names(B_map)) %>%
  mutate(cond = factor(B_map[method], levels = B_order)) %>%
  arrange(cond) %>%
  select(cond, cliff_cov)

B_fills <- c("none"        = "grey85",
             "var."        = "grey68",
             "AD"          = "grey55",
             "rough."      = "grey20",
             "var.+\nrough." = "grey40")

pB <- ggplot(barB, aes(x = cond, y = cliff_cov, fill = cond)) +
  geom_col(colour = "black", linewidth = 0.3, width = 0.74) +
  geom_hline(yintercept = 0.90, linetype = "dashed", colour = "black", linewidth = 0.5) +
  scale_fill_manual(values = B_fills, guide = "none") +
  scale_y_continuous(limits = c(0.80, 0.93), oob = scales::oob_squish,
                     expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Cliff coverage by conditioning",
       x = NULL,
       y = "coverage on activity cliffs") +
  base_theme +
  theme(axis.text.x = element_text(size = rel(0.85)))

fig <- (pA + pB) +
  plot_layout(widths = c(1.55, 1)) +
  plot_annotation(
    tag_levels = "A",
    title = paste0("Roughness, not tree variance or the applicability domain, is the ",
                   "per-compound scale that keeps coverage valid on cliffs"),
    theme = theme(plot.title = element_text(face = "bold", size = 12, hjust = 0))
  ) &
  theme(plot.tag = element_text(face = "bold", size = 13))

W <- 10; H <- 5.2
ggsave(png_path, fig, width = W, height = H, dpi = 300)

cat("saved:", png_path, "\n")
cat(sprintf("size: %.1f x %.1f in @300dpi -> %d x %d px\n",
            W, H, round(W * 300), round(H * 300)))

cat("\n-- Panel A spot-check (cov by method,qbin) --\n")
print(curveA %>% filter(method %in% c("roughness", "standard"), qbin %in% c(0, 4)) %>%
        select(method, qbin, cov) %>% arrange(method, qbin), n = Inf)
cat("\n-- Panel B spot-check (cliff_cov per bar) --\n")
print(barB)
