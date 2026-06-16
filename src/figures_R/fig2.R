#!/usr/bin/env Rscript
# fig2.R -- color-independent ggplot2 version of manuscript Figure 2
# (saved as paper/figures/figure5_enrichment.png to match the manuscript reference).
#
# Reproduces src/enrichment.py exactly: two panels of recall (% target compounds
# captured) vs % of compounds flagged as low-confidence, 4 series each
# (roughness, applicability domain, uncertainty/RF variance, random).
# All plotted values are read from results/fig2_curves.csv, which is emitted by
# `python3 src/enrichment.py` -- no statistics are re-derived here.
#
# Every series is distinguished by BOTH linetype and point shape so the figure
# stays unambiguous in pure grayscale; Okabe-Ito color is added only for screen.

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(patchwork)
})

# --- locate repo root relative to this script -------------------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg)) normalizePath(dirname(file_arg)) else getwd()
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

csv_path <- file.path(ROOT, "results", "fig2_curves.csv")
png_path <- file.path(ROOT, "paper", "figures", "figure5_enrichment.png")
stopifnot(file.exists(csv_path))

curves <- read_csv(csv_path, show_col_types = FALSE)

# --- tidy labels (keep the same content/order as enrichment.py) -------------
# Panel titles in the CSV carry the "A "/"B " prefix from matplotlib; map to
# clean facet-style titles. patchwork's tag_levels="A" supplies the A/B tags.
panel_levels <- c(
  "A  Catching high-error predictions" = "Catching high-error predictions",
  "B  Catching activity cliffs"        = "Catching activity cliffs"
)
curves <- curves %>%
  mutate(
    panel_clean = factor(panel_levels[panel], levels = unname(panel_levels)),
    frac_pct    = frac * 100,
    recall_pct  = recall * 100
  )

# Series order = order they appear in enrichment.py's scorer lists.
# "roughness" covers both nbr_disp (panel A) and SALI-density (panel B): it is
# the structure-only flag, so we collapse the two roughness labels into one
# legend entry "roughness (structure)" while keeping the per-panel values.
series_recode <- function(m) {
  dplyr::case_when(
    grepl("^roughness", m)        ~ "roughness (structure)",
    m == "applicability domain"   ~ "applicability domain",
    m == "uncertainty (RF var)"   ~ "uncertainty (RF var)",
    m == "random"                 ~ "random",
    TRUE ~ m
  )
}
series_order <- c("roughness (structure)", "applicability domain",
                  "uncertainty (RF var)", "random")
curves <- curves %>%
  mutate(series = factor(series_recode(method), levels = series_order))

# --- encodings: linetype + shape (grayscale-safe) + Okabe-Ito (screen) ------
lt_vals  <- c("roughness (structure)" = "solid",
              "applicability domain"  = "22",
              "uncertainty (RF var)"  = "dotted",
              "random"                = "dotdash")
sh_vals  <- c("roughness (structure)" = 16,   # filled circle
              "applicability domain"  = 17,   # filled triangle
              "uncertainty (RF var)"  = 15,   # filled square
              "random"                = 4)     # cross
# Okabe-Ito subset (added only on top of the grayscale-safe encoding)
col_vals <- c("roughness (structure)" = "#0072B2",
              "applicability domain"  = "#000000",
              "uncertainty (RF var)"  = "#009E73",
              "random"                = "#D55E00")

base_theme <- theme_bw(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = rel(1.0)),
    legend.title     = element_text(size = rel(0.85)),
    legend.position  = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = scales::alpha("white", 0.85),
                                     colour = NA),
    legend.key.width = unit(1.4, "lines")
  )

make_panel <- function(d, ptitle) {
  ggplot(d, aes(frac_pct, recall_pct,
                linetype = series, shape = series, colour = series)) +
    # diagonal reference line (random expectation), as in enrichment.py
    geom_abline(slope = 1, intercept = 0, linetype = "dotted",
                colour = "grey55", linewidth = 0.4) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.4, stroke = 0.9) +
    scale_linetype_manual(values = lt_vals, name = "flagging signal") +
    scale_shape_manual(values = sh_vals,   name = "flagging signal") +
    scale_colour_manual(values = col_vals, name = "flagging signal") +
    scale_x_continuous(limits = c(0, 50), breaks = seq(0, 50, 10),
                       expand = expansion(mult = c(0.01, 0.03))) +
    scale_y_continuous(limits = c(0, NA),
                       expand = expansion(mult = c(0.01, 0.05))) +
    labs(title = ptitle,
         x = "% of compounds flagged as low-confidence",
         y = "% of target compounds captured (recall)") +
    base_theme
}

pA <- make_panel(filter(curves, panel_clean == "Catching high-error predictions"),
                 "Catching high-error predictions")
pB <- make_panel(filter(curves, panel_clean == "Catching activity cliffs"),
                 "Catching activity cliffs")

combined <- (pA + pB) +
  plot_layout(guides = "keep") +
  plot_annotation(
    tag_levels = "A",
    title = paste0("Operational triage: a structure-only roughness flag captures ",
                   "more problem compounds\nthan applicability domain or uncertainty"),
    theme = theme(plot.title = element_text(face = "bold", size = rel(1.0)))
  )

W <- 10; H <- 5.4
ggsave(png_path, combined, width = W, height = H, dpi = 300)

# --- report -----------------------------------------------------------------
cat(sprintf("wrote %s\n", png_path))
# read PNG pixel dims straight from the IHDR chunk (no extra package needed)
con <- file(png_path, "rb"); on.exit(close(con))
raw_hdr <- readBin(con, "raw", n = 24)
pxw <- sum(as.integer(raw_hdr[17:20]) * 256^(3:0))
pxh <- sum(as.integer(raw_hdr[21:24]) * 256^(3:0))
cat(sprintf("PNG pixel dims: %d x %d (W x H)\n", pxw, pxh))

chk <- curves %>%
  filter(abs(frac - 0.20) < 1e-9) %>%
  arrange(panel_clean, series) %>%
  transmute(panel = as.character(panel_clean),
            series = as.character(series),
            `recall@20%` = sprintf("%.1f%%", recall_pct))
cat("\nspot-check recall@20% flagged (should match enrichment.py stdout):\n")
print(as.data.frame(chk), row.names = FALSE)
