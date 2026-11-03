"""Python counterpart of Schmate/src/main.cpp. Edit MODEL_PATH below, then run."""

from pathlib import Path
import math
import sys

import schmate as sm

# Application configuration lives in code, including the model selection.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/all-MiniLM-L6-v2.F32.gguf"
INDEX_DIRECTORY = Path(__file__).resolve().parents[1] / "demo-indexes"
INITIAL_INDEX = "default"
THREADS = 4
DEBUG = False

HELP = """Commands:
  use <name>
  append <sentence>
  appendid <sid> <sentence>
  ingest <file_path>                  one nonempty UTF-8 line per input
  search <query>
  showfull <query>
  knn [k] <query>
  radius [minScore] <query>
  relative [alpha] <query>
  adaptive [alpha] [minN] [lookahead] [gapDelta] <query>
  epsilon [value] <query>
  delete <label> [shard]
  undelete <label> [shard]
  delete_addr <address> [shard]
  undelete_addr <address> [shard]
  merge | flush | clear
  shard_count [name]
  reconstruct_label <label>
  reconstruct_sid <sid>
  set <key> <value>
  list keys | show config | show version
  help | quit
"""


def leading_options(payload, conversions, defaults):
    """Consume optional numeric arguments; preserve the remaining query text."""
    rest = payload.lstrip()
    values = list(defaults)
    for index, convert in enumerate(conversions):
        first, separator, tail = rest.partition(" ")
        try:
            value = convert(first)
        except ValueError:
            break
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Search options must be finite")
        if convert is int and value < 0:
            raise ValueError("Integer search options cannot be negative")
        values[index] = value
        rest = tail.lstrip() if separator else ""
    if not rest:
        raise ValueError("A search query is required")
    return values, rest


class Console:
    def __init__(self, manager, index=INITIAL_INDEX, output=None):
        self.manager, self.current = manager, index
        self.output = output or sys.stdout
        manager.use(index)

    def say(self, *args):
        print(*args, file=self.output)

    def results(self, results, full=False):
        if not results:
            self.say(" - (no results)")
        for result in results:
            text = (self.manager.get_text(self.current, result, full_sentence=True)
                    if full else result.text)
            self.say(f" - [score={result.score:.6f}, sid={result.sentence_id}, "
                     f"label={result.label}, shard={result.shard}, "
                     f"tokens=[{result.token_start},{result.token_end}]] {text}")
            if self.manager.config.debug:
                self.say(f"   file=[{result.file_start},{result.file_end}]")

    def execute(self, line):
        line = line.strip()
        if not line:
            return True
        command, _, payload = line.partition(" ")
        payload = payload.lstrip()
        if command in {"quit", "exit"}:
            return False
        if command == "help":
            self.say(HELP)
        elif command == "use":
            self.manager.use(payload)
            self.current = payload
            self.say("Switched to index:", payload)
        elif command == "append":
            self.manager.append(self.current, payload)
        elif command == "appendid":
            sid, text = payload.split(" ", 1)
            self.manager.append(self.current, text, sentence_id=int(sid))
            self.say("Appended with sid=" + sid)
        elif command == "ingest":
            count = 0
            with Path(payload).expanduser().open(encoding="utf-8") as source:
                for text in source:
                    text = text.rstrip("\r\n")
                    if text.strip():
                        self.manager.append(self.current, text)
                        count += 1
            self.say(f"Processed {count} inputs.")
        elif command == "search":
            self.results(self.manager.search(self.current, payload))
        elif command == "showfull":
            self.results(self.manager.knn(self.current, payload), full=True)
        elif command in {"knn", "radius", "relative", "adaptive", "epsilon"}:
            specs = {
                "knn": ([int], [0]), "radius": ([float], [-1.0]),
                "relative": ([float], [-1.0]), "epsilon": ([float], [-1.0]),
                "adaptive": ([float, int, int, float], [-1.0, 0, 0, -1.0]),
            }
            values, query = leading_options(payload, *specs[command])
            method = getattr(self.manager, "epsilon_search" if command == "epsilon" else command)
            self.results(method(self.current, query, *values))
        elif command in {"delete", "undelete", "delete_addr", "undelete_addr"}:
            arguments = payload.split()
            if not 1 <= len(arguments) <= 2:
                raise ValueError("Expected an ID and optional shard number")
            identity = int(arguments[0])
            shard = int(arguments[1]) if len(arguments) == 2 else 0
            method = {"delete": "remove", "undelete": "undelete",
                      "delete_addr": "delete_by_address", "undelete_addr": "undelete_by_address"}[command]
            getattr(self.manager, method)(self.current, identity, shard)
            self.say(command + " executed")
        elif command in {"merge", "flush", "clear"}:
            getattr(self.manager, command)(self.current)
            self.say(command + " completed")
        elif command == "shard_count":
            name = payload or self.current
            self.say(f'Index "{name}" shard count: {self.manager.shard_count(name)}')
        elif command in {"reconstruct_label", "reconstruct_sid"}:
            self.say(getattr(self.manager, command)(self.current, int(payload)))
        elif command == "set":
            key, value = payload.split(" ", 1)
            self.manager.set_config(key, value)
            self.say(f"{key}={value}")
        elif line == "list keys":
            self.say(" ".join(sm.Config.keys()))
        elif line == "show config":
            self.say(str(self.manager.config))
        elif line == "show version":
            self.say("Schmate:", sm.schmate_revision)
            self.say("bert.cpp:", sm.bert_revision)
            self.say("ggml:", sm.ggml_revision)
        else:
            raise ValueError("Unknown command. Type 'help'.")
        return True


def run(model_path=MODEL_PATH, directory=INDEX_DIRECTORY, *, input_stream=None, output=None):
    output = output or sys.stdout
    config = sm.Config()
    config.model_name = str(model_path)
    config.debug = DEBUG
    config.bert_n_threads = THREADS
    # Follow the standalone C++ demo's model/storage detection.
    storage, _ = sm.model_quantization(str(model_path))
    if storage != sm.StorageType.FLOAT32:
        config.storage_type = storage
        config.quantization = sm.Quantization.NONE
    with sm.EmbeddingModel(str(model_path), threads=THREADS) as model:
        with sm.IndexManager(model, config, directory=str(directory)) as manager:
            console = Console(manager, output=output)
            console.say("Interactive mode. Model:", model.model_name)
            console.say(HELP)
            stream = input_stream or sys.stdin
            while True:
                if stream.isatty():
                    print(f"[{console.current}]> ", end="", file=output, flush=True)
                line = stream.readline()
                if not line:
                    break
                try:
                    if not console.execute(line):
                        break
                except (ValueError, RuntimeError, OSError, OverflowError, TypeError) as error:
                    console.say("Error:", error)


if __name__ == "__main__":
    run()
