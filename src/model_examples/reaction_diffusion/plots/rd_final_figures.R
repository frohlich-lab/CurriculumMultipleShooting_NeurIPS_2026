# Manuscript figure for the reaction-diffusion benchmark (final 100-seed run): rd_benchmark.pdf.
#
# Matches the submitted manuscript (Fig. 2) and the R code its figures were made
# with (sciml_ode_training/rsrc, parse_best_new):
#   * strategies Adam, CL, CMS, MS in alphabetical order (no Adam+BFGS arm in this benchmark), with
#     the manuscript's Okabe-Ito colours;
#   * "Interpolation MSE" (training window) and "Extrapolation MSE" (validation window), log10 axes;
#   * main figure = one column per model, interpolation on top and extrapolation below, panel title
#     = model name, brackets comparing CMS with every other strategy;
#   * Wilcoxon rank-sum tests, Holm-corrected per model, shown as ns / * / ** / ***;
#   * runs with MSE >= 1e2 are dropped from the boxplots (as parse_best_new's loss_filter_val); the
#     tests use all 100 runs.
#
# Input: plots/<experiment>/data/rd_final_traces.csv from export_final_for_r.py.
# Usage (from the repo root, R with ggplot2 / ggsignif / patchwork, e.g. R/4.2.2-foss-2022b):
#   Rscript src/model_examples/reaction_diffusion/plots/rd_final_figures.R [experiment]
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(ggsignif)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
experiment <- if (length(args) >= 1) args[1] else "reaction_diffusion_final_19102026"
plot_root <- file.path("plots", experiment)

col_use <- c("Adam" = "#E69F00", "Adam BFGS" = "#56B4E9", "CL" = "#009E73",
             "CMS" = "#CC79A7", "MS" = "#0072B2")
tag_levels <- c("Adam", "CL", "CMS", "MS")
model_title <- c("UDE" = "Reaction-diffusion UDE model",
                 "NODE" = "Reaction-diffusion NeuralODE")

# Printed size: Fig. 2 is placed at 5.45 x 2.6 in for three model columns (~1.8 in per column), so
# two columns at the same scale are ~3.7 x 2.6 in. Everything below is set for that size, so the
# PDF goes in at 100% and its text matches Fig. 2 (~5 pt ticks, ~6.5 pt axis and panel titles).
FIG_WIDTH <- 3.7
FIG_HEIGHT <- 2.6
BASE_SIZE <- 6.3
loss_filter_val <- 1e2

my_theme <- theme_bw(base_size = BASE_SIZE) +
  theme(legend.position = "bottom",
        axis.title = element_text(color = "grey10"),
        plot.title = element_text(color = "grey10", size = BASE_SIZE + 0.5,
                                  margin = margin(0, 0, 2, 0)),
        plot.title.position = "plot",
        plot.tag = element_text(size = BASE_SIZE + 1),
        axis.title.x = element_blank(),
        plot.margin = margin(2, 2, 0, 0),
        panel.grid = element_line(linewidth = 0.2),
        axis.ticks = element_line(linewidth = 0.2),
        panel.border = element_rect(linewidth = 0.3),
        plot.subtitle = element_text(color = "grey30"))

df <- read_csv(file.path(plot_root, "data", "rd_final_traces.csv"), show_col_types = FALSE) |>
  mutate(tag = factor(tag, levels = tag_levels))
df_end <- df |>
  group_by(model, tag, start_guess) |>
  filter(epoch == max(epoch)) |>
  ungroup()

save_fig <- function(p, dir_save, name, width, height) {
  path <- file.path(dir_save, str_c(name, ".pdf"))
  ggsave(path, p, width = width, height = height)
  message("SAVED ", path)
}

stars <- function(p) {
  case_when(p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ "ns")
}

# CMS vs every other strategy, rank-sum, Holm over all of one model's tests (both metrics).
others <- setdiff(tag_levels, "CMS")
tests <- bind_rows(lapply(c("UDE", "NODE"), function(m) {
  d <- filter(df_end, model == m)
  bind_rows(lapply(c("loss_train", "loss_val"), function(metric) {
    bind_rows(lapply(others, function(o) {
      x <- d[[metric]][d$tag == "CMS"]
      y <- d[[metric]][d$tag == o]
      tibble(model = m, metric = metric, other = o,
             median_cms = median(x), median_other = median(y),
             p = wilcox.test(x, y, exact = FALSE)$p.value)
    }))
  }))
})) |>
  group_by(model) |>
  mutate(p_holm = p.adjust(p, method = "holm"), label = stars(p_holm)) |>
  ungroup()

box_panel <- function(m, metric, ylab, title = NULL, tag = NULL) {
  d <- filter(df_end, model == m, .data[[metric]] < loss_filter_val)
  t <- filter(tests, model == m, metric == !!metric)
  # flat brackets stacked above the data, Adam (furthest from CMS) on top, as in Fig. 2
  top <- log10(max(d[[metric]]))
  step <- 0.09 * (top - log10(min(d[[metric]])))  # spacing scales with the panel's range
  comparisons <- lapply(t$other, function(o) c(o, "CMS"))
  y_pos <- top + step * rev(seq_along(comparisons))
  ggplot(d, aes(x = tag, y = .data[[metric]], fill = tag)) +
    geom_boxplot(linewidth = 0.25, outlier.size = 0.35) +
    geom_signif(comparisons = comparisons, annotations = t$label, y_position = y_pos,
                tip_length = 0, textsize = 1.9, vjust = 0.3, size = 0.2, color = "grey20") +
    scale_fill_manual(values = col_use) +
    scale_y_log10(expand = expansion(mult = c(0.05, 0.10))) +  # headroom for the top label
    labs(x = "", y = ylab, title = title, tag = tag) +
    my_theme +
    theme(legend.position = "none")
}

model_column <- function(m, tag) {
  box_panel(m, "loss_train", "Interpolation MSE", model_title[[m]], tag) /
    box_panel(m, "loss_val", "Extrapolation MSE")
}

# --- main figure (Fig. 2 layout): panel letters on the top row only, one per model ---
main <- model_column("UDE", "a") | model_column("NODE", "b")
save_fig(main, plot_root, "rd_benchmark", FIG_WIDTH, FIG_HEIGHT)

# --- tables: medians (as in App. A) and the tests behind the brackets ---
write_csv(df_end |>
            group_by(model, tag) |>
            summarise(median_interpolation_mse = median(loss_train),
                      mad_interpolation_mse = mad(loss_train),
                      median_extrapolation_mse = median(loss_val),
                      mad_extrapolation_mse = mad(loss_val),
                      n_dropped_from_box = sum(loss_train >= loss_filter_val |
                                                 loss_val >= loss_filter_val),
                      .groups = "drop"),
          file.path(plot_root, "rd_summary_table.csv"))
write_csv(tests |> mutate(metric = recode(metric, loss_train = "interpolation",
                                          loss_val = "extrapolation")),
          file.path(plot_root, "rd_wilcoxon_cms_vs_others.csv"))
print(as.data.frame(tests), digits = 3)
