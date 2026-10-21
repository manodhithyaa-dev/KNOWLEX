# tfidf_rag.R : TF-IDF retrieval + Ollama answer generation
# One-time setup:
#   install.packages("httr")
#   ollama pull llama3.2:3b        # or llama3.2:1b
# Run from the project folder:
#   Rscript tfidf_rag.R "what are the assumptions of linear regression"
#   Rscript tfidf_rag.R            # interactive mode

library(httr)

setwd("/home/manodhithyaa-c-s/Desktop/PROBABILITY PROJECT")

KNOWLEDGE_DIR <- "knowledge"
CHUNK_WORDS   <- 150
OVERLAP_WORDS <- 30
TOP_K         <- 3

OLLAMA_URL    <- "http://localhost:11434/api/chat"
OLLAMA_MODEL  <- "llama3.2:3b"    # change to "llama3.2:1b" if you pulled that one
TEMPERATURE   <- 0.2

stopwords <- c("a","an","the","and","or","but","if","of","to","in","on","at","by",
               "for","with","as","is","are","was","were","be","been","this","that","these",
               "those","it","its","from","into","than","then","so","such","can","will","not",
               "we","you","i","he","she","they","them","their","our","your","which","who",
               "what","when","where","how","do","does","did","has","have","had","also")

tokenize <- function(text) {
  text   <- tolower(text)
  text   <- gsub("[^a-z0-9]+", " ", text)
  tokens <- strsplit(trimws(text), "\\s+")[[1]]
  tokens[nchar(tokens) > 1 & !(tokens %in% stopwords)]
}

# ---- 1. Load and chunk documents -------------------------------------------
load_chunks <- function(dir) {
  files <- list.files(dir, pattern = "\\.md$", full.names = TRUE)
  if (length(files) == 0) stop("No .md files found in '", dir, "'")
  chunks <- list()
  step <- CHUNK_WORDS - OVERLAP_WORDS
  for (f in files) {
    txt   <- paste(readLines(f, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    words <- strsplit(trimws(txt), "\\s+")[[1]]
    if (length(words) == 0) next
    starts <- seq(1, length(words), by = step)
    for (s in starts) {
      e <- min(s + CHUNK_WORDS - 1, length(words))
      chunks[[length(chunks) + 1]] <- list(
        source = basename(f),
        text   = paste(words[s:e], collapse = " ")
      )
      if (e == length(words)) break
    }
  }
  chunks
}

# ---- 2. Build TF-IDF index --------------------------------------------------
build_index <- function(chunks) {
  docs_tokens <- lapply(chunks, function(ch) tokenize(ch$text))
  vocab <- sort(unique(unlist(docs_tokens)))
  N <- length(docs_tokens)
  
  tf <- matrix(0, nrow = N, ncol = length(vocab), dimnames = list(NULL, vocab))
  for (i in seq_len(N)) {
    tab <- table(docs_tokens[[i]])
    if (length(tab) > 0)
      tf[i, names(tab)] <- as.numeric(tab) / sum(tab)
  }
  
  df  <- colSums(tf > 0)
  idf <- log((1 + N) / (1 + df)) + 1
  
  tfidf <- sweep(tf, 2, idf, `*`)
  list(chunks = chunks, vocab = vocab, idf = idf, tfidf = tfidf)
}

# ---- 3. Retrieve top-k chunks via cosine similarity --------------------------
cosine_sim <- function(M, v) {
  num <- as.vector(M %*% v)
  den <- sqrt(rowSums(M^2)) * sqrt(sum(v^2))
  ifelse(den == 0, 0, num / den)
}

retrieve <- function(index, query, k = TOP_K) {
  tokens <- tokenize(query)
  tokens <- tokens[tokens %in% index$vocab]
  if (length(tokens) == 0) return(NULL)
  
  q <- setNames(numeric(length(index$vocab)), index$vocab)
  tab <- table(tokens)
  q[names(tab)] <- as.numeric(tab) / sum(tab)
  q <- q * index$idf
  
  scores <- cosine_sim(index$tfidf, q)
  top <- order(scores, decreasing = TRUE)[seq_len(min(k, length(scores)))]
  data.frame(
    rank   = seq_along(top),
    score  = round(scores[top], 4),
    source = vapply(index$chunks[top], function(c) c$source, ""),
    text   = vapply(index$chunks[top], function(c) c$text, ""),
    stringsAsFactors = FALSE
  )
}

# ---- 4. Answer generation with Ollama ----------------------------------------
build_prompt <- function(query, res) {
  context <- paste(
    sprintf("[%d] (source: %s)\n%s", res$rank, res$source, res$text),
    collapse = "\n\n"
  )
  paste0(
    "Context:\n", context,
    "\n\nQuestion: ", query,
    "\n\nAnswer using only the context above. Cite the passage numbers like [1], [2]. ",
    "If the context does not contain the answer, say you don't know."
  )
}

generate_answer <- function(query, res) {
  body <- list(
    model    = OLLAMA_MODEL,
    stream   = FALSE,
    options  = list(temperature = TEMPERATURE),
    messages = list(
      list(role = "system",
           content = "You are a careful assistant that answers strictly from the provided context."),
      list(role = "user", content = build_prompt(query, res))
    )
  )
  
  resp <- tryCatch(
    POST(OLLAMA_URL, body = body, encode = "json", timeout(300)),
    error = function(e) {
      stop("Could not reach Ollama at ", OLLAMA_URL,
           ". Is it running? (try: ollama serve)\n", conditionMessage(e))
    }
  )
  
  if (status_code(resp) != 200) {
    stop("Ollama error ", status_code(resp), ": ",
         content(resp, as = "text", encoding = "UTF-8"),
         "\nHave you run: ollama pull ", OLLAMA_MODEL, " ?")
  }
  content(resp, as = "parsed")$message$content
}

# ---- 5. Output + pipeline -----------------------------------------------------
show_sources <- function(res) {
  cat("\n--- Retrieved passages ---\n")
  for (i in seq_len(nrow(res))) {
    cat(sprintf("\n[%d] score=%.4f  source=%s\n", res$rank[i], res$score[i], res$source[i]))
    cat(strwrap(res$text[i], width = 90), sep = "\n")
  }
  cat("\n")
}

ask <- function(index, query) {
  res <- retrieve(index, query)
  if (is.null(res) || nrow(res) == 0) {
    cat("No matching terms found in the knowledge base.\n"); return(invisible())
  }
  cat("\nGenerating answer with ", OLLAMA_MODEL, "...\n", sep = "")
  ans <- tryCatch(generate_answer(query, res),
                  error = function(e) { cat("Error:", conditionMessage(e), "\n"); NULL })
  if (!is.null(ans)) {
    cat("\n=== Answer ===\n")
    cat(strwrap(ans, width = 90), sep = "\n")
  }
  show_sources(res)
}

# ---- 6. Main -----------------------------------------------------------------
chunks <- load_chunks(KNOWLEDGE_DIR)
index  <- build_index(chunks)
cat(sprintf("Indexed %d chunks, %d terms from '%s'\n",
            length(chunks), length(index$vocab), KNOWLEDGE_DIR))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  ask(index, paste(args, collapse = " "))
} else {
  con <- file("stdin")
  open(con)
  repeat {
    cat("Query (empty to quit): ")
    q <- readLines(con, n = 1)
    if (length(q) == 0 || trimws(q) == "") break
    ask(index, q)
  }
  close(con)
}
