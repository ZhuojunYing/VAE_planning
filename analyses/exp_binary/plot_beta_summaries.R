#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(i, default) {
  if (length(args) >= i && nzchar(args[[i]])) args[[i]] else default
}

beta_arg <- get_arg(1, "0.01,0.1,1.0,10.0,100.0")
lambda_arg <- get_arg(2, "1.0")
alpha_arg <- get_arg(3, "0.0")
opportunity_arg <- get_arg(4, "0.0")
input_dir <- get_arg(5, "outputs/simulations")
results_dir <- get_arg(6, "results")
tree_size <- as.integer(get_arg(7, "2"))
input_type <- get_arg(8, "uniform")

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

beta_values <- trimws(strsplit(beta_arg, ",")[[1]])
seeds <- 1:5

value_candidates <- function(x) {
  x_chr <- as.character(x)
  x_num <- suppressWarnings(as.numeric(x_chr))
  candidates <- x_chr
  if (!is.na(x_num)) {
    candidates <- c(
      candidates,
      sprintf("%.1f", x_num),
      sprintf("%.2f", x_num),
      format(x_num, scientific = FALSE, trim = TRUE)
    )
  }
  unique(candidates)
}

simulation_path <- function(lambda_value, alpha_value, beta_value, opportunity_value, seed) {
  lambda_candidates <- value_candidates(lambda_value)
  alpha_candidates <- value_candidates(alpha_value)
  beta_candidates <- value_candidates(beta_value)
  opportunity_candidates <- value_candidates(opportunity_value)

  for (lambda_candidate in lambda_candidates) {
    for (alpha_candidate in alpha_candidates) {
      for (beta_candidate in beta_candidates) {
        for (opportunity_candidate in opportunity_candidates) {
          file_names <- c(
            sprintf(
              "lambda_%s_alpha_%s_beta_%s_opportunity_%s_seed_%d_%dn_%s.csv",
              lambda_candidate, alpha_candidate, beta_candidate, opportunity_candidate, seed, tree_size, input_type
            )
          )
          for (file_name in file_names) {
            file_path <- file.path(input_dir, file_name)
            if (file.exists(file_path)) {
              return(file_path)
            }
          }
        }
      }
    }
  }

  NA_character_
}

read_seed_file <- function(beta_value, seed) {
  file_path <- simulation_path(lambda_arg, alpha_arg, beta_value, opportunity_arg, seed)
  if (is.na(file_path)) {
    warning(sprintf("Missing simulation file for beta=%s seed=%d", beta_value, seed))
    return(NULL)
  }

  dat <- read.csv(file_path, stringsAsFactors = FALSE)
  dat$beta <- beta_value
  dat$seed <- seed
  dat$source_file <- file_path
  dat
}

all_data <- do.call(
  rbind,
  unlist(
    lapply(beta_values, function(beta_value) {
      lapply(seeds, function(seed) read_seed_file(beta_value, seed))
    }),
    recursive = FALSE
  )
)

if (is.null(all_data) || nrow(all_data) == 0) {
  stop("No simulation CSVs were found. Check beta/lambda/alpha values and input_dir.")
}

if ("opportunity_cost" %in% names(all_data)) {
  requested_opportunity <- suppressWarnings(as.numeric(opportunity_arg))
  if (!is.na(requested_opportunity)) {
    all_data <- all_data[
      abs(suppressWarnings(as.numeric(all_data$opportunity_cost)) - requested_opportunity) < 1e-8,
      ,
      drop = FALSE
    ]
  }

  if (nrow(all_data) == 0) {
    stop(sprintf("No rows matched opportunity_cost=%s.", opportunity_arg))
  }
}

as_logical_col <- function(x) {
  if (is.logical(x)) {
    return(x)
  }
  tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y", "stop")
}

build_current_stop_data <- function(dat) {
  reward_cols <- grep("^expanded_reward_t[0-9]+$", names(dat), value = TRUE)
  reward_cols <- reward_cols[order(as.integer(sub("^expanded_reward_t", "", reward_cols)))]
  stop_cols <- grep("^stop_t[0-9]+$", names(dat), value = TRUE)
  stop_cols <- stop_cols[order(as.integer(sub("^stop_t", "", stop_cols)))]

  if (length(reward_cols) == 0 || length(stop_cols) == 0) {
    stop(
      paste(
        "Cannot compute timestep-specific stop probabilities.",
        "Expected expanded_reward_t* and stop_t* columns. The loaded files only contain:",
        paste(names(dat), collapse = ", ")
      )
    )
  }

  n_steps <- min(length(reward_cols), length(stop_cols))
  if (n_steps < 2) {
    stop("Need at least two timesteps to compute P(stop at t+1 | reward observed at t).")
  }

  trial_id_cols <- intersect(c("beta", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, reward_cols[seq_len(n_steps)], stop_cols[seq_len(n_steps)]))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    for (t in seq_len(n_steps - 1)) {
      reward_t <- suppressWarnings(as.numeric(trial_data[[reward_cols[[t]]]][[i]]))
      if (!is.na(reward_t)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          seed = trial_data$seed[[i]],
          graph = trial_data$graph[[i]],
          reward_timestep = t,
          decision_timestep = t + 1,
          reward = reward_t,
          stop_current = as_logical_col(trial_data[[stop_cols[[t + 1]]]][[i]]),
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    stop("No observed rewards were found before a current stop decision.")
  }

  do.call(rbind, rows)
}

stop_data <- build_current_stop_data(all_data)

stop_summary <- aggregate(
  stop_current ~ beta + reward_timestep + decision_timestep + reward,
  data = stop_data,
  FUN = mean
)
names(stop_summary)[names(stop_summary) == "stop_current"] <- "p_stop_current"

stop_counts <- aggregate(
  stop_current ~ beta + reward_timestep + decision_timestep + reward,
  data = stop_data,
  FUN = length
)
names(stop_counts)[names(stop_counts) == "stop_current"] <- "n"
stop_summary <- merge(
  stop_summary,
  stop_counts,
  by = c("beta", "reward_timestep", "decision_timestep", "reward")
)

if ("MI" %in% names(all_data)) {
  all_data$MI_value <- all_data$MI
} else if ("MI_cost" %in% names(all_data)) {
  all_data$MI_value <- all_data$MI_cost
} else {
  stop("Expected an MI or MI_cost column for the average V vs MI plot.")
}

v_mi_seed <- aggregate(
  cbind(V, MI_value) ~ beta + seed,
  data = all_data,
  FUN = mean
)
v_mi_summary <- aggregate(
  cbind(V, MI_value) ~ beta,
  data = v_mi_seed,
  FUN = mean
)

build_kl_summary <- function(dat) {
  kl_cols <- grep("^kl_d_t[0-9]+$", names(dat), value = TRUE)
  kl_cols <- kl_cols[order(as.integer(sub("^kl_d_t", "", kl_cols)))]

  if (length(kl_cols) == 0) {
    stop(
      paste(
        "Expected kl_d_t* columns for KL plotting, but none were found.",
        "Please re-run simulate.py with the updated model outputs."
      )
    )
  }

  trial_id_cols <- intersect(c("beta", "seed", "graph"), names(dat))
  trial_cols <- unique(c(trial_id_cols, kl_cols))
  trial_data <- unique(dat[, trial_cols, drop = FALSE])

  rows <- list()
  row_i <- 1
  for (i in seq_len(nrow(trial_data))) {
    for (kl_col in kl_cols) {
      timestep <- as.integer(sub("^kl_d_t", "", kl_col))
      kl_value <- suppressWarnings(as.numeric(trial_data[[kl_col]][[i]]))
      if (!is.na(kl_value)) {
        rows[[row_i]] <- data.frame(
          beta = trial_data$beta[[i]],
          seed = trial_data$seed[[i]],
          graph = trial_data$graph[[i]],
          timestep = timestep,
          kl_d = kl_value,
          stringsAsFactors = FALSE
        )
        row_i <- row_i + 1
      }
    }
  }

  if (length(rows) == 0) {
    stop("KL columns were found, but all kl_d_t* values are NA.")
  }

  kl_data <- do.call(rbind, rows)
  aggregate(
    kl_d ~ beta + timestep,
    data = kl_data,
    FUN = mean
  )
}

kl_summary <- build_kl_summary(all_data)

beta_levels <- beta_values[beta_values %in% unique(all_data$beta)]
if (length(beta_levels) == 0) {
  beta_levels <- unique(all_data$beta)
}

palette_cols <- grDevices::hcl.colors(max(3, length(beta_levels)), palette = "Dark 3")
beta_cols <- setNames(palette_cols[seq_along(beta_levels)], beta_levels)

expand_range <- function(x, pad = 0.5) {
  x_range <- range(x, finite = TRUE)
  if (!all(is.finite(x_range))) {
    return(c(0, 1))
  }
  if (identical(x_range[[1]], x_range[[2]])) {
    return(c(x_range[[1]] - pad, x_range[[2]] + pad))
  }
  x_range
}

stop_pdf <- file.path(
  results_dir,
  sprintf(
    "stop_probability_%s_lambda_%s_alpha_%s_opportunity_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, opportunity_arg, tree_size
  )
)

plot_stop_panel <- function(reward_timestep) {
  panel_data <- stop_summary[
    stop_summary$reward_timestep == reward_timestep,
    ,
    drop = FALSE
  ]

  decision_timestep <- reward_timestep + 1
  panel_title <- if (reward_timestep == 1) {
    "Stop at timestep 2\nafter reward at timestep 1"
  } else if (reward_timestep == 2) {
    "Stop at timestep 3\nafter reward at timestep 2"
  } else {
    sprintf(
      "Stop at timestep %d\nafter reward at timestep %d",
      decision_timestep,
      reward_timestep
    )
  }

  if (nrow(panel_data) == 0) {
    plot(
      NA,
      xlim = c(-0.1, 1.1),
      ylim = c(0, 1),
      xlab = "Observed reward",
      ylab = "P(stop at current timestep)",
      main = panel_title,
      xaxt = "n"
    )
    axis(1, at = c(0, 1))
    grid()
    text(
      0.5,
      0.5,
      sprintf(
        "No stop_t%d decision exists\nafter reward_t%d",
        decision_timestep,
        reward_timestep
      ),
      cex = 0.9
    )
    return(invisible(NULL))
  }

  plot(
    NA,
    xlim = expand_range(panel_data$reward, pad = 0.1),
    ylim = c(0, 1),
    xlab = "Observed reward",
    ylab = "P(stop at current timestep)",
    main = panel_title,
    xaxt = "n"
  )
  axis(1, at = sort(unique(panel_data$reward)))
  grid()

  for (beta_value in beta_levels) {
    beta_dat <- panel_data[panel_data$beta == beta_value, , drop = FALSE]
    beta_dat <- beta_dat[order(beta_dat$reward), , drop = FALSE]
    if (nrow(beta_dat) > 0) {
      lines(
        beta_dat$reward,
        beta_dat$p_stop_current,
        type = "b",
        pch = 19,
        lwd = 2,
        col = beta_cols[[beta_value]]
      )
    }
  }
}

pdf(stop_pdf, width = 11, height = 5)
old_par <- par(mfrow = c(1, 2), mar = c(4.5, 4.5, 4, 1))
plot_stop_panel(1)
plot_stop_panel(2)
par(old_par)
legend(
  "topright",
  inset = c(0.02, 0.02),
  legend = paste("beta", beta_levels),
  col = beta_cols[beta_levels],
  pch = 19,
  lwd = 2,
  bty = "n"
)
dev.off()

v_mi_pdf <- file.path(
  results_dir,
  sprintf(
    "average_V_vs_MI_%s_lambda_%s_alpha_%s_opportunity_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, opportunity_arg, tree_size
  )
)

pdf(v_mi_pdf, width = 6.5, height = 5.5)
plot(
  v_mi_summary$MI_value,
  v_mi_summary$V,
  xlim = expand_range(v_mi_summary$MI_value, pad = 0.05),
  ylim = expand_range(v_mi_summary$V, pad = 0.05),
  xlab = "Average MI",
  ylab = "Average V",
  main = sprintf(
    "Average V vs MI (lambda=%s, alpha=%s, opportunity=%s)",
    lambda_arg, alpha_arg, opportunity_arg
  ),
  pch = 19,
  cex = 1.3,
  col = beta_cols[as.character(v_mi_summary$beta)]
)
grid()
text(
  v_mi_summary$MI_value,
  v_mi_summary$V,
  labels = paste("beta", v_mi_summary$beta),
  pos = 4,
  cex = 0.8
)
legend(
  "topright",
  legend = paste("beta", beta_levels),
  col = beta_cols[beta_levels],
  pch = 19,
  bty = "n"
)
dev.off()

kl_pdf <- file.path(
  results_dir,
  sprintf(
    "average_kl_d_%s_lambda_%s_alpha_%s_opportunity_%s_%dn.pdf",
    input_type, lambda_arg, alpha_arg, opportunity_arg, tree_size
  )
)

pdf(kl_pdf, width = 7, height = 5.5)
plot(
  NA,
  xlim = expand_range(kl_summary$timestep, pad = 0.1),
  ylim = expand_range(kl_summary$kl_d, pad = 0.05),
  xlab = "Timestep",
  ylab = "Average kl_d",
  main = sprintf(
    "Average kl_d by timestep (lambda=%s, alpha=%s, opportunity=%s)",
    lambda_arg, alpha_arg, opportunity_arg
  ),
  xaxt = "n"
)
axis(1, at = sort(unique(kl_summary$timestep)))
grid()

for (beta_value in beta_levels) {
  beta_dat <- kl_summary[kl_summary$beta == beta_value, , drop = FALSE]
  beta_dat <- beta_dat[order(beta_dat$timestep), , drop = FALSE]
  if (nrow(beta_dat) > 0) {
    lines(
      beta_dat$timestep,
      beta_dat$kl_d,
      type = "b",
      pch = 19,
      lwd = 2,
      col = beta_cols[[beta_value]]
    )
  }
}

legend(
  "topright",
  legend = paste("beta", beta_levels),
  col = beta_cols[beta_levels],
  pch = 19,
  lwd = 2,
  bty = "n"
)
dev.off()

message("Wrote: ", stop_pdf)
message("Wrote: ", v_mi_pdf)
message("Wrote: ", kl_pdf)
