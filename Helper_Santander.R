######################################
# Helper Function Santander Challenge #
#######################################

install_these_packages <- function(listPackages) {
    new.packages <- listPackages[!(listPackages %in% installed.packages()[,"Package"])]
    if(length(new.packages)) install.packages(new.packages, dependencies = TRUE, repos = "https://cran.cnr.berkeley.edu/")
}

santander_libs <- function(loadLibrary = TRUE) {
    # Data wrangling, manipulation, processing and machine learning
    packages <- c("irlba", "dtplyr", "tidyverse", "parallelMap", "data.table", "mlr") # dbplyr
    install_these_packages(packages)
    
    if (loadLibrary) l <- lapply(packages, function(t) suppressMessages(library(t, character.only = TRUE)))
}

col_duplicated <- function(data) {
    # Determina el número y el % de datos duplicados por columnas en un data frame. Retorna la tabla.
    data <- data.frame(data)
    namesColumn <- names(data)
    listDuplicated <- sapply(namesColumn, function(t) sum(duplicated(data[t])))
    tableDuplicated <- data.frame(columns = names(listDuplicated), duplicates = listDuplicated,
                                  percent = round(listDuplicated/nrow(data),4), row.names = NULL, stringsAsFactors = F)
    return(tableDuplicated)
}

table_na <- function(data) {
    # Entrega una tabla que cuantifica cuantos NA hay por columna y el % con respecto al total
    data <- data.frame(data)
    names_column <- names(data)
    list_na <- sapply(names_column, function(t) sum(is.na(data[t])))
    missing_data <- data.frame(column = names(data), numberNA = list_na, row.names = NULL, stringsAsFactors = F)
    missing_data$percent <- round(missing_data$numberNA/nrow(data)*100, 4)
    missing_dataOrder <- missing_data[order(missing_data$numberNA, decreasing = T),]
    return(missing_dataOrder)
}

table_0 <- function(x) {
    col_type <- sapply(x, class)
    col_type <- col_type %in% c("numeric", "integer", "integer64")
    x <- x[, col_type]
    x <- as.matrix(x)
    x_sum <- apply(x, 2, function(t) sum(t == 0))
    spars <- data.frame(column = names(x_sum), count_0 = x_sum, row.names = NULL, stringsAsFactors = FALSE)
    spars$count_1 <- nrow(x) - spars$count_0
    spars$percent <- round(spars$count_0/nrow(x), 4)
    spars_order <- spars[order(spars$count_0, decreasing = T),]
    return(spars_order)
} 

rm_sparse_terms <- function(x, upper_bound = 1) {
    spars <- table_0(x)
    columns.for.elimination <- spars$column[which(spars$percent >= upper_bound)] # column names
    columns.for.elimination <- which(dimnames(x)[[2]] %in% columns.for.elimination) # column numbers in the dataset
    if (length(columns.for.elimination) > 0) x <- x[, -c(columns.for.elimination)] 
    return(x)
}

# Interna de funciones, para la evaluación no estandar de variables en funciones.
# Permite usar agregar variables en funciones sin tener que ponerle comillas.
# Util para crear funciones que puedan recibir pipes (%>%)
# Autor: Google
col_name <- function (x, default = stop("Please supply column name", call. = FALSE)) {
    if (is.character(x))
        return(x)
    if (identical(x, quote(expr = )))
        return(default)
    if (is.name(x))
        return(as.character(x))
    if (is.null(x))
        return(x)
    stop("Invalid column specification", call. = FALSE)
}

gghist <- function(data, x, tlog = F, type = "count", xlab = NULL, bin = 50, fill = "dodgerblue", f = mean) {
    # Genera un histograma para las variables continuas.
    # Tiene la opción de ser un histograma dpor frecuencia (type = "count") o por densidad (type = "density").
    # Se puede aplicar una transformación logaritmica si los datos están muy desviados (tlog = TRUE).
    # Adicionalmente se puede pasar el número de bins y nombre del eje X.
    # Adicionalmente dibuja encima el promedio de la distribución.
    .x <- col_name(substitute(x))
    if (tlog == F) {
        xcol <- data[[.x]]
        xlab <- toupper(ifelse(is.null(xlab), .x, xlab))
        p <- ggplot(data, aes(x = xcol))
    } else {
        xcol <- log10(data[[.x]] + 1)
        xlab <- toupper(ifelse(is.null(xlab), paste(.x, "(escala logaritmica)"),
                               paste(xlab, "(escala logaritmica)")))
        p <-  ggplot(data, aes(x= log10(data[[.x]] + 1)))
    }
    
    
    
    if (type == "count") {
        p <- p + geom_histogram(aes(y=..count..), bins = bin, colour="black", fill = fill, alpha = 0.7) +
            geom_vline(aes(xintercept  = f(xcol, na.rm = T)), linetype="dashed", size=1.1, col = "red") +
            xlab(xlab) + ylab("FRECUENCIA")
    } else if (type == "density") {
        p <- p + geom_histogram(aes(y=..density..), bins = bin, colour="black", fill = fill, alpha = 0.7) +
            geom_density(alpha=.4, colour = "black", size = 1, fill = "salmon") +
            geom_vline(aes(xintercept  = f(xcol, na.rm = T)), linetype="dashed", size=1.1, col = "red") +
            xlab(xlab) + ylab("DENSIDAD")
    }
    return(p)
}

# Genera un gráfico de barras horizontal que cuantifica la cantidad de una variable categórica.
# Te genera un gráfico que te muestra la distribución de una variable categórica.
# Incluye el % de cada variable en el gráfico. Te retorna también la tabla por el cual fue construido,
# por si es necesario. Si no se asigna a una variable se imprime en consola.
# Autor: Fabián
ggbar <- function(data, category, highcol = 'black', top = NULL, title = NULL) {
    .category <- col_name(substitute(category))
    
    table <- data %>% data.frame %>% count_(.category, sort = TRUE) %>%
        mutate(perc = round(n/sum(n)*100, 2))
    
    if(!is.null(top) & is.numeric(top)) {
        table <- head(table, top)
    } else if (!is.null(top) & !is.numeric(top)) {
        stop("ERROR en 'top': Ingrese un valor númerico")
    }
    
    p <- ggplot(table, aes(x = reorder(table[[.category]], n), y = n, fill = n)) + geom_bar(stat = 'identity') +
        coord_flip() +
        scale_fill_gradient(low = "lightgrey", high = highcol) +
        labs(y = "FRECUENCIA", x = "") + ylim(0, max(table$n * 1.15)) +
        theme(text = element_text(size = 16), legend.position = "none",
              plot.margin = unit(c(1,2,1,1), "cm")) +
        geom_text(aes(label = paste0(n, " (", scales::percent(perc/100), ")")), hjust = -0.3, size = 3.5, fontface = "bold")
    
    if(!is.null(title)) {
        p <- p + ggtitle(title)
    }
    
    print(table)
    return(p)
}

# Genera un gráfico de torta de una variable categórica.
# Cuantifica una variable, la ordena y la muestra con sus respectivos porcentajes.
# Esta función se recomienda para cuando hay igual o menos de 10 categorías.
# Si hay más se recomienda el uso de ggbar. Sobre 10 variables el relleno es blanco y no se muestra su porcentaje.
# La paleta de colores se basa en los de tableau, tienes a tu disposición:
# tableau20 (default)
# tableau10medium
# tableau10light
# colorblind10
# trafficlight
# purplegray12
# bluered12
# greenorange12
# cyclic
# Autor: Fabián
ggpie <- function(data, category, color = "tableau20", sentido = 1) {
    .category <- col_name(substitute(category))
    if (length(unique(data[[.category]])) > 10) {
        message("Sobre 10 variables categóricas detectadas, para vizualizar mejor usa 'ggbar'")
    }
    
    table <- data %>% data.frame %>% count_(.category) %>%
        mutate(perc = round(n/sum(n)*100, 2)) %>%
        arrange(perc)
    
    y <- table$perc
    
    breaks <- if (length(y) > 5) {
        (cumsum(y) - y/2)[length(y):(length(y) - 8)]
    } else {
        (cumsum(y) - y/2)[length(y):1]
    }
    
    ylabels <- if (length(y) > 5) {
        table$n[length(y):(length(y) - 8)]
    } else {
        table$n[length(y):1]
    }
    
    alpha <- if (length(y) > 5) {
        rev(c(rep(1, 5), rep(0, length(y) - 5)))
    } else {
        1
    }
    
    if (sentido == -1) {
        fill <- reorder(table[[.category]], y)
        cy <- (cumsum(rev(y)) - rev(y)/2)
        label <- rev(paste0(y, "%"))
        breaks <- if (length(y) > 5) {
            (cumsum(rev(y)) - rev(y)/2)[length(y):(length(y) - 8)]
        } else {
            (cumsum(rev(y)) - rev(y)/2)[length(y):1]
        }
        ylabels <- rev(ylabels)
    } else {
        fill <- reorder(table[[.category]], rev(y))
        cy <- (cumsum(y) - y/2)
        label <- paste0(y, "%")
    }
    
    p <- ggplot(table, aes(x=1, y = y, fill = fill)) +
        geom_bar(stat = "identity", color = "black", lwd = 0.5) +
        coord_polar(theta = "y", direction = sentido) +
        ggthemes::scale_fill_tableau(color) +
        theme(axis.ticks=element_blank(),
              axis.title=element_blank(),
              axis.text.y=element_blank(),
              axis.text.x=element_text(color='black', face = "bold"),
              panel.grid.major.y=element_line(color = "black")) +
        scale_y_continuous(
            breaks = breaks,
            labels = ylabels) +
        ggrepel::geom_text_repel(aes(y = cy, label = label),
                                 alpha = alpha, size=4, color = "white", fontface = "bold")  +
        labs(fill = .category) +
        theme(legend.title = element_text(face = "bold"), 
              text = element_text(size = 16),
              panel.background = element_rect(fill = "white"))
    
    print(table)
    return(p)
}

log10p <- function(x) {
    log10(x + 1)
}

exp10p <- function(x) {
    10^(x) - 1
}

RMSLE_fun <- function(task, model, pred, feats, extra.args) {
    MLmetrics::RMSLE(y_pred = exp10p(pred$data$response), y_true = exp10p(pred$data$truth))
}

RMSLE_kaggle <- mlr::makeMeasure(
    id = "RMSLE_kaggle", name = "Compute the RMSLE with transformation",
    properties = "regr",
    minimize = TRUE, best = 0, worst = Inf, 
    fun = RMSLE_fun
)

preProc_data <- function(df, add_Xcols = TRUE, logTransform = TRUE, sparsity = 1, ceroVariance = TRUE, removeDuplicates = TRUE) {
    
    if (add_Xcols) {
        message("\nAdjusting columns names...")
        names(df) <- paste0("x", tolower(names(df))) # Add X before each columns to format columns correctly
        names(df) <- gsub("xid", "id", names(df)) # return id and target to normal
        names(df) <- gsub("xtarget", "target", names(df))
    }

    # Remove Sparse data 
    if (sparsity > 0) {
        message("Removing columns with over ", sparsity*100, " % sparsity")
        prev_ncol <- ncol(df)
        df <- rm_sparse_terms(df, upper_bound = sparsity) 
        cat("A total of", prev_ncol - ncol(df), "columns were removed!\n")
    }
    
    # Remove cero variance columns
    if (ceroVariance) {
        message("Removing cero variance columns...")
        prev_ncol <- ncol(df)
        col_var <- sapply(df[-1], var) > 0
        col_var <- c(XID = TRUE, col_var)
        df <- df[, which(col_var)]
        cat("A total of", prev_ncol - ncol(df), "columns were removed!\n")
    }
    
    # Remove duplicated Columns
    if (removeDuplicates) {
        message("Removing duplicated columns...")
        prev_ncol <- ncol(df)
        df <- as.data.frame(as.list(df)[!duplicated(as.list(df))], stringsAsFactors = FALSE) # 5 duplicated columns
        cat("A total of", prev_ncol - ncol(df), "columns were removed!\n")
    }
    
    # LogTransform all the data 
    if (logTransform) {
        message("Transforming dataframe with log10p...")
        setDT(df)
        cols <- which(!grepl("id", names(df))) # extract the colnames to log-transform
        df[, (cols) := lapply(.SD, log10p), .SDcols = cols] # applies log10(x + 1) trasformation 
    }
    
    cat("Done!\n")
    return(as.tibble(df))
}


row_aggregates <- function(df) {
    df_res <- df %>% select(id)
    dt <- df %>% select(starts_with("x")) %>% as.matrix()
    dt[dt == 0] <- NA # zeros to NA's to calculate row aggregates not considering zero rows 
    
    cat("\nRow Sums...")
    df_res$row_sums_zv <- matrixStats::rowSums2(dt, na.rm = TRUE); cat("Complete!")
    cat("\nRow means...")
    df_res$row_means_zv <- matrixStats::rowMeans2(dt, na.rm = TRUE); cat("Complete!")
    cat("\nRow medians...")
    df_res$row_median_zv <- matrixStats::rowMedians(dt, na.rm = TRUE); cat("Complete!")
    cat("\nRow Standard Deviation...")
    df_res$row_sds_zv <- matrixStats::rowSds(dt, na.rm = TRUE); cat("Complete!")
    cat("\nRow Max...")
    df_res$row_max_zv <- matrixStats::rowMaxs(dt, na.rm = TRUE); cat("Complete!")
    cat("\nRow Min...")
    df_res$row_min_zv <- matrixStats::rowMins(dt, na.rm = TRUE); cat("Complete!")
    cat("\nZero Sum...")
    df_res$sum_zeros <- matrixStats::rowCounts(dt, value = NA); cat("Complete!")
    cat("\nValue Sum...")
    dt[is.na(dt)] <- 0 # returning NAs to zero
    dt[dt > 0] <- 1 # turning every non-zero value to one, easily to count
    df_res$sum_values <- matrixStats::rowCounts(dt, value = ); cat("Complete!")
    df_res[is.na(df_res)] <- 0
    df_res$row_max_zv[is.infinite(df_res$row_max_zv)] <- 0
    df_res$row_min_zv[is.infinite(df_res$row_min_zv)] <- 0
    return(df_res)
}


principal_component_fit <- function(train_df, nComp, center = TRUE, scale = TRUE) {
    # perform fast principal component analysis
    df_target <- train_df %>% select(id)
    dt <- train_df %>% select(starts_with("x"))
    pc_irlba <- prcomp_irlba(dt, n = nComp, center = center, scale. = scale) 
    df_pc <- cbind(df_target, pc_irlba$x)
    df_pc$id <- as.character(df_pc$id)
    return(list(train_pc = df_pc, pc_irlba = pc_irlba))
}

principal_component_transform <- function(test_df, pca_object) {
    df_target <- test_df %>% select(id)
    dt <- test_df %>% select(starts_with("x"))
    pca_test <- predict(pca_object, newdata = dt)
    test_pc <- cbind(df_target, pca_test)
    test_pc$id <- as.character(test_pc$id)
    return(test_pc)
}

tSVD_fit <- function(train_df, n_vectors, maxit = 600, center = NULL, scale = NULL) {
    df_target <- train_df %>% select(id)
    dt <- train_df %>% select(starts_with("x")) %>% as.matrix
    tSVD_res <- irlba(t(dt), nv = n_vectors, maxit = maxit, center = center, scale = scale)
    svd_vect <- data.frame(tSVD_res$v, stringsAsFactors = FALSE)
    names(svd_vect) <- paste0("SVD", 1:n_vectors)
    train_svd <- cbind(df_target, svd_vect)
    train_svd$id <- as.character(train_svd$id)
    return(list(train_svd = train_svd, tSVD_res = tSVD_res))
}

tSVD_transform <- function(test_df, SVD_object) {
    df_target <- test_df %>% select(id)
    dt <- test_df %>% select(starts_with("x")) %>% as.matrix
    sigma_inverse <- 1 / SVD_object$d
    u_transpose <- t(SVD_object$u)
    dt <- t(sigma_inverse * u_transpose %*% t(dt))
    dt <- as.data.frame(dt)
    names(dt) <- paste0("SVD", 1:ncol(dt))
    test_svd <- cbind(df_target, dt)
    test_svd$id <- as.character(test_svd$id)
    return(test_svd)
}

replicateFeaturesInTest <- function(train, test) {
    df_train_names <- getTaskData(train) %>% names
    df_test <- getTaskData(test)
    df_test[, df_train_names]
}


