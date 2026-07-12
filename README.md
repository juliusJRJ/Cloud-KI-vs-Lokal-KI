# Cloud-KI-vs-Lokal-KI
## Klassifikation mehrsprachiger Support-Tickets: Prototyp, Cloud-LLM und lokales LLM im Vergleich

Dieses Repository enthält das Begleitmaterial zu dem Projekt im Rahmen des Kurses Wissensmanagement, die drei unterschiedliche Ansätze zur automatischen Klassifikation mehrsprachiger Kundenservice-Tickets einander gegenüberstellt: ein klassisches Machine-Learning-Verfahren auf Basis von Sentence-Embeddings und XGBoost, einen cloudbasierten Ansatz über die Anthropic-API (Claude) sowie einen vollständig lokal auf eigener Hardware ausgeführten Ansatz über die Laufzeitumgebung Ollama. Alle drei Verfahren werden entlang derselben drei Dimensionen bewertet, nämlich der Qualität der Klassifikation, den damit verbundenen Kosten und der Latenz, wodurch sich trotz der grundverschiedenen technischen Umsetzung ein direkter Vergleich ziehen lässt.

## Aufbau des Repositories

Das Repository besteht aus drei eigenständigen Jupyter-Notebooks, die jeweils einen der drei Ansätze abbilden und unabhängig voneinander ausgeführt werden können, sofern die jeweiligen Voraussetzungen erfüllt sind.

- **`Prototyp_Customer_Support_Classification.ipynb`**: Klassischer ML-Ansatz, bei dem die Ticket-Texte mittels vortrainierter Sentence-Embeddings in numerische Repräsentationen überführt und anschließend von einem XGBoost-Klassifikator einer Kategorie zugeordnet werden. Das Notebook dokumentiert dabei auch die iterative Weiterentwicklung von einem kompakten englischsprachigen Embedding-Modell zu einem leistungsfähigeren mehrsprachigen Modell.
- **`Cloud_Approach__Claude__LLM_Classification.ipynb`**: Die Klassifikation wird vollständig an ein über die Anthropic-API angesprochenes Sprachmodell delegiert. Es findet kein eigenes Training statt, die Bewertung erfolgt anhand von Tokenverbrauch, Latenz und Klassifikationsgüte.
- **`Local_Approach_LLM_Classification.ipynb`**: Funktional identischer Aufbau wie das Cloud-Notebook, jedoch wird die Inferenz über die lokale Laufzeitumgebung Ollama auf der eigenen Hardware ausgeführt, sodass die Ticketdaten das Gerät zu keinem Zeitpunkt verlassen. Als Kostengröße tritt hier die aufgewendete Rechenzeit an die Stelle des Token-Entgelts.

Damit die drei Ansätze überhaupt vergleichbar sind, teilen sich alle Notebooks dieselbe Datengrundlage, dieselbe Schema-Bereinigung der Zielkategorien, denselben stratifizierten Stichprobenzug und einen inhaltlich identischen Klassifikationsprompt.

## Datengrundlage

Verwendet wird der mehrsprachige Kaggle-Datensatz *multilingual-customer-support-tickets* (`tobiasbueck/multilingual-customer-support-tickets`), aus dem gezielt die rund 20.000 Zeilen umfassende multilinguale Variante geladen wird. Jedes Notebook bezieht die Datei automatisch über die Bibliothek `kagglehub`, ein manuelles Herunterladen ist also nicht notwendig.

Vor der eigentlichen Klassifikation wird in allen drei Notebooks dieselbe Bereinigung des Kategorieschemas vorgenommen: Die ursprünglich getrennt geführten Kategorien *IT Support* und *Technical Support* werden zu einer gemeinsamen Kategorie *Technical and IT Support* zusammengeführt, da sich beide inhaltlich stark überschneiden, und die schwach besetzte, für die Aufgabenstellung irrelevante Kategorie *Human Resources* wird vollständig entfernt. Im Ergebnis verbleiben acht Zielkategorien, auf denen sämtliche drei Ansätze bewertet werden.

## Voraussetzungen

Empfohlen wird eine Python-Umgebung ab Version 3.11, beispielsweise über ein virtuelles Environment (`.venv`) oder alternativ Google Colab, wobei Letzteres nur für das Prototyp- und das Cloud-Notebook geeignet ist. Das lokale LLM-Notebook setzt einen auf demselben Rechner laufenden Ollama-Dienst voraus und lässt sich daher nicht in einer entfernten Umgebung wie Colab betreiben.

### Gemeinsame Python-Abhängigkeiten

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Zusätzlich für den ML-Prototyp

```bash
pip install kagglehub sentence-transformers xgboost
```

Da `sentence-transformers` auf PyTorch aufsetzt, empfiehlt sich auf Rechnern ohne dedizierte Grafikkarte die Installation einer CPU-only-Variante, um unnötige CUDA-Abhängigkeiten zu vermeiden:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Für den Zugriff auf den Kaggle-Datensatz über `kagglehub` kann je nach Konfiguration ein Kaggle-Konto samt API-Token erforderlich sein. Sollte der Download fehlschlagen, ist ein solches Token unter dem eigenen Kaggle-Profil zu erzeugen und entweder als Datei `kaggle.json` im Verzeichnis `~/.kaggle` oder über die Umgebungsvariablen `KAGGLE_USERNAME` und `KAGGLE_KEY` bereitzustellen.

### Zusätzlich für den Cloud-Ansatz

```bash
pip install anthropic
```

Das Notebook benötigt einen gültigen API-Schlüssel der Anthropic-API, der aus Sicherheitsgründen nicht im Notebook selbst hinterlegt, sondern aus der Umgebungsvariable `ANTHROPIC_API_KEY` gelesen wird. In Google Colab kann der Schlüssel komfortabel über das Secrets-Panel unter genau diesem Namen abgelegt werden. Zu beachten ist, dass jede Ausführung des Notebooks reale Kosten bei Anthropic verursacht, da für die Stichprobe von 200 Tickets tatsächliche API-Aufrufe getätigt werden.

### Zusätzlich für den lokalen LLM-Ansatz

```bash
pip install ollama
```

Vorausgesetzt wird eine lokale Installation von [Ollama](https://ollama.com), deren Dienst standardmäßig unter `http://localhost:11434` erreichbar ist. Vor der ersten Ausführung muss das gewünschte Modell einmalig über das Terminal bezogen werden, etwa mit:

```bash
ollama pull gemma3n:e4b
```

Der im Notebook über die Variable `MODEL` hinterlegte Modellname muss dabei mit einem lokal bereits geladenen Modell übereinstimmen. Im Rahmen der Arbeit wurden neben `gemma3n:e4b` auch `qwen2.5:3b` und `llama3.2:3b` getestet; ein Wechsel des Modells erfordert lediglich das Anpassen dieser einen Variable sowie das vorherige Laden des jeweiligen Modells per `ollama pull`. Da die Inferenz vollständig auf der lokalen CPU erfolgt, ist bei Geräten mit wenig Arbeitsspeicher (ab etwa 8 GB) die Verwendung kompakter, quantisierter Modelle zu empfehlen.

## Wichtige Konfigurationsparameter

Für die Reproduzierbarkeit der Ergebnisse sind einige Parameter zwischen den Notebooks bewusst konsistent gehalten und sollten bei eigenen Anpassungen nicht unabhängig voneinander verändert werden. Die Stichprobengröße ist einheitlich auf `SAMPLE_SIZE = 200` festgelegt, und die stratifizierte Ziehung verwendet in allen drei Notebooks denselben Zufallsstartwert `random_state=42`, damit alle drei Ansätze exakt dieselben Tickets bewerten. Die oben beschriebene Schema-Bereinigung ist ebenfalls identisch umgesetzt und muss vor jeder Stichprobenziehung angewendet werden, da sie sowohl die zulässige Kategorienliste als auch die Klassenverteilung der Stichprobe beeinflusst.

Im Prototyp-Notebook ist außerdem zu beachten, dass die am Ende dokumentierte Gittersuche zur Hyperparameter-Optimierung auf der Testmenge zu einem schlechteren Ergebnis führte als die vorangehende Konfiguration. Als maßgebliches Endergebnis des Prototyps gilt daher die Ausbaustufe vor der Gittersuche, bestehend aus dem mehrsprachigen Embedding-Modell `paraphrase-multilingual-mpnet-base-v2` in Kombination mit einem klassengewichteten XGBoost-Klassifikator.

## Ergebnisse im Überblick

Die folgenden Werte fassen die zentralen Befunde der Arbeit zusammen und dienen der Einordnung; die vollständige Herleitung sowie die klassenweise Aufschlüsselung finden sich in den jeweiligen Notebooks.

| Ansatz | Accuracy | Makro-F1 |
| --- | --- | --- |
| ML-Prototyp (Sentence-Embeddings + XGBoost) | 0,57 | 0,48 |
| Cloud-LLM (Claude Sonnet über API) | 0,41 | – |
| Lokales LLM, bestes Modell (`gemma3n:e4b`) | 0,43–0,46 (je nach Gerät) | – |

Bemerkenswert ist dabei, dass der vergleichsweise einfache und ressourcenschonende ML-Prototyp beide LLM-basierten Ansätze in der Klassifikationsgüte übertrifft, obwohl weder das Cloud- noch das lokale Sprachmodell eigens für diese Aufgabe trainiert wurden.

### Verwendete lokale Hardware

Die für den lokalen Ansatz berichteten Rechenzeiten und Latenzen wurden auf zwei unterschiedlich ausgestatteten Laptops ohne dedizierte Grafikkarte gemessen. Diese Angaben sind für die reine Ausführung der Notebooks nicht erforderlich, dienen jedoch der Einordnung der im lokalen Notebook erzielten Zeitmessungen.

| | Gerät 1 | Gerät 2 |
| --- | --- | --- |
| Prozessor | Intel Core i3-1115G4 (11. Gen.) | Intel Core i5-12450H (12. Gen.) |
| Arbeitsspeicher | 8 GB, 3733 MT/s | 16 GB, 4267 MT/s |
| Grafikkarte | keine | keine |
| Betriebssystem | Windows 11 | Windows 11 |

## Ausführung

Jedes Notebook lässt sich nach Installation der jeweils benötigten Abhängigkeiten von oben nach unten durchlaufen. Die Zellen sind so aufgebaut, dass zunächst der Datensatz geladen und bereinigt, anschließend die Stichprobe gezogen und danach die eigentliche Klassifikation samt Messung von Qualität, Kosten und Latenz durchgeführt wird. Da im Cloud-Notebook reale API-Kosten anfallen und im lokalen Notebook je nach Modell und Hardware mit mehreren Minuten bis Stunden Laufzeit für die Stichprobe zu rechnen ist, empfiehlt sich vor einer vollständigen Ausführung ein kurzer Blick in die jeweiligen Konfigurationszellen.

