#!/usr/bin/env Rscript
# Grayscale-safe ggplot2 reproduction of manuscript Figure 2 enrichment curves.

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg)) normalizePath(dirname(file_arg)) else getwd()
ROOT <- normalizePath(file.path(script_dir, "..", ".."))

csv_path <- file.path(ROOT, "results", "fig2_curves.csv")
png_path <- file.path(ROOT, "paper", "figures", "figure5_enrichment.png")
stopifnot(file.exists(csv_path))

curves <- read_csv(csv_path, show_col_types = FALSE)

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

lt_vals  <- c("roughness (structure)" = "solid",
              "applicability domain"  = "22",
              "uncertainty (RF var)"  = "dotted",
              "random"                = "dotdash")
sh_vals  <- c("roughness (structure)" = 16,
              "applicability domain"  = 17,
              "uncertainty (RF var)"  = 15,
              "random"                = 4)
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

cat(sprintf("wrote %s\n", png_path))
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
