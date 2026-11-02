# Manuscript figures for the reaction-diffusion benchmark (final 100-seed run).
#
# Matches the submitted manuscript (Fig. 1-2 and App. A.7) and the R code its figures were made
# with (sciml_ode_training/rsrc, parse_best_new):
#   * strategies Adam, CL, CMS, MS in alphabetical order (no Adam+BFGS arm in this benchmark), with
#     the manuscript's Okabe-Ito colours;
#   * "Interpolation MSE" (training window) and "Extrapolation MSE" (validation window), log10 axes;
#   * main figure = one column per model, interpolation on top and extrapolation below, panel title
#     = model name, brackets comparing CMS with every other strategy;
#   * Wilcoxon rank-sum tests, Holm-corrected per model, shown as ns / * / ** / ***;
#   * runs with MSE >= 1e2 are dropped from the boxplots (as parse_best_new's loss_filter_val) and
#     counted in the bad-run figures instead; the tests use all 100 runs.
# Supplementary: median + IQR traces over epochs (curriculum end dashed, per facet: CL ends at epoch
# 458 with 12 stages, CMS at 400 with 5) and bad-run counts.
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

BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0
loss_filter_val <- 1e2

my_theme <- theme_bw(base_size = 14) +
  theme(legend.position = "bottom",
        axis.title = element_text(color = "grey10"),
        plot.title = element_text(color = "grey10", size = 14),
        plot.subtitle = element_text(color = "grey30"))

df <- read_csv(file.path(plot_root, "data", "rd_final_traces.csv"), show_col_types = FALSE) |>
  mutate(tag = factor(tag, levels = tag_levels))
df_end <- df |>
  group_by(model, tag, start_guess) |>
  filter(epoch == max(epoch)) |>
  ungroup()

save_fig <- function(p, dir_save, name, width, height) {
  if (!dir.exists(dir_save)) dir.create(dir_save, recursive = TRUE)
  for (ext in c("svg", "pdf")) ggsave(file.path(dir_save, str_c(name, ".", ext)), p,
                                      width = width, height = height)
  ggsave(file.path(dir_save, str_c(name, ".png")), p, width = width, height = height, dpi = 150)
  message("SAVED ", file.path(dir_save, str_c(name, ".svg")))
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

box_panel <- function(m, metric, ylab, title = NULL) {
  d <- filter(df_end, model == m, .data[[metric]] < loss_filter_val)
  t <- filter(tests, model == m, metric == !!metric)
  # flat brackets stacked above the data, Adam (furthest from CMS) on top, as in Fig. 2
  top <- log10(max(d[[metric]]))
  step <- 0.09 * (top - log10(min(d[[metric]])))  # spacing scales with the panel's range
  comparisons <- lapply(t$other, function(o) c(o, "CMS"))
  y_pos <- top + step * rev(seq_along(comparisons))
  ggplot(d, aes(x = tag, y = .data[[metric]], fill = tag)) +
    geom_boxplot() +
    geom_signif(comparisons = comparisons, annotations = t$label, y_position = y_pos,
                tip_length = 0, textsize = 4.5, vjust = 0.2, color = "grey20") +
    scale_fill_manual(values = col_use) +
    scale_y_log10(expand = expansion(mult = c(0.05, 0.10))) +  # headroom for the top label
    labs(x = "", y = ylab, title = title) +
    my_theme +
    theme(legend.position = "none")
}

model_column <- function(m) {
  box_panel(m, "loss_train", "Interpolation MSE", model_title[[m]]) /
    box_panel(m, "loss_val", "Extrapolation MSE")
}

# --- main figure (Fig. 2 layout) and the per-model panels ---
main <- wrap_elements(model_column("UDE")) | wrap_elements(model_column("NODE"))
main <- main + plot_annotation(tag_levels = "a")
save_fig(main, plot_root, "rd_benchmark", BASE_WIDTH * 1.6, BASE_HEIGHT * 1.6)
for (m in c("UDE", "NODE")) {
  save_fig(box_panel(m, "loss_train", "Interpolation MSE", model_title[[m]]),
           file.path(plot_root, m), "box_interpolation", BASE_WIDTH, BASE_HEIGHT)
  save_fig(box_panel(m, "loss_val", "Extrapolation MSE", model_title[[m]]),
           file.path(plot_root, m), "box_extrapolation", BASE_WIDTH, BASE_HEIGHT)
}

# --- supplementary: traces and bad runs per model ---
for (m in c("UDE", "NODE")) {
  d <- filter(df, model == m)
  dir_save <- file.path(plot_root, m)
  n_starts <- n_distinct(d$start_guess)
  multistarts <- str_c(model_title[[m]], " (", n_starts, " multistarts)")
  df_traces <- d |>
    group_by(tag, epoch) |>
    summarise(across(c(loss_train, loss_val),
                     list(median = median,
                          q25 = ~ quantile(.x, 0.25, na.rm = TRUE),
                          q75 = ~ quantile(.x, 0.75, na.rm = TRUE))),
              .groups = "drop")
  df_cl_end <- d |> filter(tag %in% c("CL", "CMS")) |> distinct(tag, n_cl_epochs)

  trace_plot <- function(metric, ylab) {
    med <- str_c(metric, "_median")
    q25 <- str_c(metric, "_q25")
    q75 <- str_c(metric, "_q75")
    y_max <- ifelse(max(df_traces[[q75]]) > 1e9, 1e9, NaN)
    ggplot(df_traces, aes(x = epoch, y = .data[[med]], color = tag, fill = tag)) +
      geom_line(linewidth = 1.5) +
      geom_ribbon(aes(ymin = .data[[q25]], ymax = .data[[q75]]), alpha = 0.2) +
      geom_vline(data = df_cl_end, aes(xintercept = n_cl_epochs), linetype = "dashed",
                 color = "grey20") +
      facet_wrap(~ tag, ncol = 4) +
      scale_y_log10(limits = c(NaN, y_max)) +
      scale_fill_manual(values = col_use) +
      scale_color_manual(values = col_use) +
      labs(x = "Epoch", y = ylab, title = multistarts) +
      my_theme +
      theme(legend.position = "none")
  }
  save_fig(trace_plot("loss_train", "Interpolation MSE"), dir_save, "trace_interpolation",
           BASE_WIDTH * 1.5, BASE_HEIGHT * 0.75)
  save_fig(trace_plot("loss_val", "Extrapolation MSE"), dir_save, "trace_extrapolation",
           BASE_WIDTH * 1.5, BASE_HEIGHT * 0.75)

  df_bad <- df_end |>
    filter(model == m) |>
    group_by(tag) |>
    summarise(bad_train = sum(loss_train >= 1e2), bad_val = sum(loss_val >= 1e2))
  bad_plot <- function(col, ylab, title) {
    ggplot(df_bad, aes(tag, .data[[col]], fill = tag)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = .data[[col]]), vjust = -0.5, size = 6, color = "grey20") +
      scale_fill_manual(values = col_use) +
      labs(x = "", y = ylab, title = title) +
      scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100),
                         expand = c(0, 0, 0.01, 0.01)) +
      my_theme +
      theme(legend.position = "none")
  }
  save_fig(bad_plot("bad_train", "Number of bad runs",
                    "Runs with interpolation MSE > 1e2"), dir_save, "bad_interpolation",
           BASE_WIDTH, BASE_HEIGHT)
  save_fig(bad_plot("bad_val", "Number of bad runs",
                    "Runs with extrapolation MSE > 1e2"), dir_save, "bad_extrapolation",
           BASE_WIDTH, BASE_HEIGHT)
}

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
