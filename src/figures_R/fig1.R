#!/usr/bin/env Rscript
# Grayscale-safe ggplot2 re-make of manuscript Figure 1 (panels A, B, C stacked vertically).

suppressPackageStartupMessages({
  library(ggplot2); library(readr); library(dplyr); library(tidyr)
  library(patchwork); library(ggrepel)
})

ROOT <- Sys.getenv("QSAR_ROOT", "")
if (!nzchar(ROOT)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(file_arg) && nzchar(file_arg)) {
    fa <- if (startsWith(file_arg, "/")) file_arg else file.path(getwd(), file_arg)
    ROOT <- dirname(dirname(dirname(fa)))
  } else {
    ROOT <- getwd()
  }
}
RESULTS  <- file.path(ROOT, "results")
FIGURES  <- file.path(ROOT, "paper", "figures")
dir.create(FIGURES, showWarnings = FALSE, recursive = TRUE)
PNG_PATH <- file.path(FIGURES, "figure1_main.png")

read_panel <- function(name) read_csv(file.path(RESULTS, name), show_col_types = FALSE)
A  <- read_panel("fig1_panelA.csv")
B  <- read_panel("fig1_panelB.csv")
Cc <- read_panel("fig1_panelC.csv")

FAM_LEVELS <- c("landscape", "a-priori", "applicability domain", "uncertainty")
FAM_GREY   <- c("landscape" = "grey20", "a-priori" = "grey45",
                "applicability domain" = "grey68", "uncertainty" = "grey85")
FAM_OI <- c("landscape" = "#000000", "a-priori" = "#0072B2",
            "applicability domain" = "#999999", "uncertainty" = "#009E73")

base_theme <- theme_bw(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title  = element_text(face = "bold", size = rel(1.0)),
        legend.title = element_text(size = rel(0.85)),
        plot.title.position = "plot")

A$family <- factor(A$family, levels = FAM_LEVELS)
Cc$family <- factor(Cc$family, levels = FAM_LEVELS)

A <- A %>% arrange(ypos) %>% mutate(label = factor(label, levels = label))
pA <- ggplot(A, aes(x = median_rho, y = label, fill = family)) +
  geom_vline(xintercept = 0, colour = "black", linewidth = 0.4) +
  geom_col(colour = "black", linewidth = 0.3, width = 0.72) +
  geom_errorbarh(aes(xmin = q25, xmax = q75), height = 0.28, colour = "black", linewidth = 0.4) +
  scale_fill_manual(values = FAM_GREY, drop = FALSE, name = "family") +
  labs(title = "What predicts where QSAR errs",
       x = expression("Spearman " * rho * " vs per-compound error (median ± IQR, 30 targets)"),
       y = NULL) +
  annotate("text", x = max(A$q75), y = 0.6, hjust = 1, vjust = 0, size = 3,
           fontface = "italic", label = "* uses the query's own activity") +
  base_theme +
  theme(legend.position = c(0.99, 0.04), legend.justification = c(1, 0),
        legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
        legend.key.size = unit(0.9, "lines"))

Bl <- B %>%
  pivot_longer(c(zero_order, partial_AD), names_to = "stage", values_to = "rho") %>%
  mutate(stage = factor(stage, levels = c("zero_order", "partial_AD"),
                        labels = c("zero-order ρ", "partial ρ\n(control: NN-sim + density)")),
         x = ifelse(grepl("^zero", stage), 0, 1),
         series = factor(label, levels = B$label))

lt_vals <- c("solid", "22", "dotted", "dotdash")[seq_len(nrow(B))]
sh_vals <- c(16, 17, 15, 4)[seq_len(nrow(B))]
oi_vals <- FAM_OI[as.character(B$family)]; names(oi_vals) <- B$label

pB <- ggplot(Bl, aes(x = x, y = rho, group = series,
                     linetype = series, shape = series, colour = series)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3, fill = "white", stroke = 0.9) +
  ggrepel::geom_text_repel(
    data = Bl %>% filter(x == 1),
    aes(label = series), nudge_x = 0.16, hjust = 0, direction = "y",
    segment.size = 0.3, segment.colour = "grey55", box.padding = 0.35,
    min.segment.length = 0, size = 3.4, show.legend = FALSE) +
  scale_x_continuous(breaks = c(0, 1),
                     labels = c("zero-order ρ", "partial ρ\n(control: NN-sim + density)"),
                     limits = c(-0.12, 1.75), expand = expansion(mult = c(0.02, 0))) +
  scale_linetype_manual(values = lt_vals, name = "construct") +
  scale_shape_manual(values = sh_vals, name = "construct") +
  scale_colour_manual(values = oi_vals, name = "construct") +
  labs(title = "Roughness survives the applicability-domain control",
       x = NULL, y = expression("Spearman " * rho)) +
  base_theme +
  theme(legend.position = "none")

Cc <- Cc %>% arrange(ypos) %>% mutate(label = factor(label, levels = label))
pC <- ggplot(Cc, aes(x = median_auc, y = label, fill = family)) +
  geom_vline(xintercept = 0.5, colour = "black", linewidth = 0.6, linetype = "dashed") +
  geom_col(colour = "black", linewidth = 0.3, width = 0.72) +
  geom_errorbarh(aes(xmin = q25, xmax = q75), height = 0.28, colour = "black", linewidth = 0.4) +
  scale_fill_manual(values = FAM_GREY, drop = FALSE, name = "family") +
  coord_cartesian(xlim = c(0.3, 0.85)) +
  labs(title = "Flagging cliffs a-priori (no activity used)",
       x = "AUC for flagging labelled cliffs (median ± IQR)", y = NULL) +
  base_theme +
  theme(legend.position = c(0.99, 0.04), legend.justification = c(1, 0),
        legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
        legend.key.size = unit(0.9, "lines"))

fig <- pA / pB / pC +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))

W <- 7.0; H <- 7.2
ggsave(PNG_PATH, fig, width = W, height = H, dpi = 300, bg = "white")
cat(sprintf("saved %s (%.1f x %.1f in @300dpi -> %d x %d px)\n",
            PNG_PATH, W, H, as.integer(W*300), as.integer(H*300)))
