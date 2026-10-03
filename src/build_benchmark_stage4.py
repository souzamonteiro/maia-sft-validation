from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.common_stage4 import (
    DATA,
    OUT,
    ROOT,
    SEED,
    normalize,
    read_jsonl,
    write_jsonl,
)


# ============================================================
# MAIA SFT VALIDATION
# STAGE 4 V4 - FROZEN ASSISTANT BENCHMARK
#
# PURPOSE
#
# Evaluate the same GPT-2 Medium model before and after
# Stage 4 response-only SFT.
#
# DESIGN
#
#   300 cases total
#
#   EN 100
#   PT 100
#   ES 100
#
# Per language:
#
#   60 objective
#   40 qualitative
#
# Objective:
#
#   arithmetic
#   comparison
#   string manipulation
#   logic
#   factual/scientific
#   code
#
# Qualitative:
#
#   explanation
#   summarization
#   rewriting
#   translation
#   coding
#   scientific reasoning
#   uncertainty
#   formatting
#   multi-turn
#
# IMPORTANT
#
# The benchmark is:
#
#   - deterministic
#   - independent of SFT data
#   - audited against all Stage 4 corpus user prompts
#   - suitable for baseline/post-SFT comparison
#
# No model is used to generate benchmark cases.
# ============================================================


BENCHMARK_DIR = DATA / "benchmark"

BENCHMARK_FILE = (
    BENCHMARK_DIR
    / "stage4-v4-benchmark.jsonl"
)

MANIFEST_FILE = (
    BENCHMARK_DIR
    / "stage4-v4-benchmark-manifest.json"
)

SEED_BENCHMARK = 20261001


# ============================================================
# HELPERS
# ============================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def message(
    role: str,
    content: str,
):
    return {
        "role": role,
        "content": content.strip(),
    }


def objective_case(
    case_id,
    language,
    category,
    prompt,
    expected,
    *,
    evaluator="normalized_exact",
    aliases=None,
):
    if aliases is None:
        aliases = []

    return {
        "id":
            case_id,

        "language":
            language,

        "kind":
            "objective",

        "category":
            category,

        "messages": [
            message(
                "user",
                prompt,
            )
        ],

        "expected": {
            "answer":
                str(expected),

            "aliases":
                [
                    str(x)
                    for x in aliases
                ],

            "evaluator":
                evaluator,
        },
    }


def qualitative_case(
    case_id,
    language,
    category,
    messages,
    criteria,
):
    return {
        "id":
            case_id,

        "language":
            language,

        "kind":
            "qualitative",

        "category":
            category,

        "messages":
            messages,

        "criteria":
            criteria,
    }


def add_objective(
    cases,
    language,
    category,
    prompt,
    expected,
    *,
    evaluator="normalized_exact",
    aliases=None,
):
    index = (
        1
        + sum(
            1
            for case in cases
            if (
                case["language"]
                == language
                and case["kind"]
                == "objective"
            )
        )
    )

    cases.append(
        objective_case(
            (
                f"{language}-obj-"
                f"{index:03d}"
            ),
            language,
            category,
            prompt,
            expected,
            evaluator=evaluator,
            aliases=aliases,
        )
    )


def add_qualitative(
    cases,
    language,
    category,
    messages,
    criteria,
):
    index = (
        1
        + sum(
            1
            for case in cases
            if (
                case["language"]
                == language
                and case["kind"]
                == "qualitative"
            )
        )
    )

    cases.append(
        qualitative_case(
            (
                f"{language}-qual-"
                f"{index:03d}"
            ),
            language,
            category,
            messages,
            criteria,
        )
    )


# ============================================================
# OBJECTIVE BENCHMARK
# ============================================================


def build_objective(
    cases,
    language,
):
    rng = random.Random(
        SEED_BENCHMARK
        + {
            "en": 1,
            "pt": 2,
            "es": 3,
        }[language]
    )

    # --------------------------------------------------------
    # Language templates
    # --------------------------------------------------------

    if language == "en":
        arithmetic_templates = {
            "add":
                "What is {a} + {b}? "
                "Answer with only the number.",

            "sub":
                "What is {a} - {b}? "
                "Answer with only the number.",

            "mul":
                "What is {a} × {b}? "
                "Answer with only the number.",
        }

        compare_template = (
            "Which number is larger: "
            "{a} or {b}? "
            "Answer with only the larger number."
        )

        lowercase_template = (
            'Convert "{text}" to lowercase. '
            "Return only the converted text."
        )

        uppercase_template = (
            'Convert "{text}" to uppercase. '
            "Return only the converted text."
        )

        length_template = (
            'How many characters are in "{text}"? '
            "Count letters only and answer "
            "with only the number."
        )

    elif language == "pt":
        arithmetic_templates = {
            "add":
                "Quanto é {a} + {b}? "
                "Responda somente com o número.",

            "sub":
                "Quanto é {a} - {b}? "
                "Responda somente com o número.",

            "mul":
                "Quanto é {a} × {b}? "
                "Responda somente com o número.",
        }

        compare_template = (
            "Qual número é maior: "
            "{a} ou {b}? "
            "Responda somente com o maior número."
        )

        lowercase_template = (
            'Converta "{text}" para letras minúsculas. '
            "Retorne somente o texto convertido."
        )

        uppercase_template = (
            'Converta "{text}" para letras maiúsculas. '
            "Retorne somente o texto convertido."
        )

        length_template = (
            'Quantos caracteres existem em "{text}"? '
            "Conte apenas as letras e responda "
            "somente com o número."
        )

    else:
        arithmetic_templates = {
            "add":
                "¿Cuánto es {a} + {b}? "
                "Responde solamente con el número.",

            "sub":
                "¿Cuánto es {a} - {b}? "
                "Responde solamente con el número.",

            "mul":
                "¿Cuánto es {a} × {b}? "
                "Responde solamente con el número.",
        }

        compare_template = (
            "¿Qué número es mayor: "
            "{a} o {b}? "
            "Responde solamente con el número mayor."
        )

        lowercase_template = (
            'Convierte "{text}" a minúsculas. '
            "Devuelve solamente el texto convertido."
        )

        uppercase_template = (
            'Convierte "{text}" a mayúsculas. '
            "Devuelve solamente el texto convertido."
        )

        length_template = (
            '¿Cuántos caracteres hay en "{text}"? '
            "Cuenta solamente las letras y responde "
            "solamente con el número."
        )

    # ========================================================
    # 1. ARITHMETIC - 15
    # ========================================================

    operations = (
        ["add"] * 5
        + ["sub"] * 5
        + ["mul"] * 5
    )

    for operation in operations:
        if operation == "add":
            a = rng.randint(
                11,
                89,
            )

            b = rng.randint(
                11,
                89,
            )

            result = a + b

        elif operation == "sub":
            a = rng.randint(
                40,
                120,
            )

            b = rng.randint(
                5,
                a - 1,
            )

            result = a - b

        else:
            a = rng.randint(
                3,
                14,
            )

            b = rng.randint(
                3,
                14,
            )

            result = a * b

        add_objective(
            cases,
            language,
            "arithmetic",
            arithmetic_templates[
                operation
            ].format(
                a=a,
                b=b,
            ),
            result,
            evaluator="numeric",
        )

    # ========================================================
    # 2. COMPARISON - 10
    # ========================================================

    for _ in range(10):
        a = rng.randint(
            -50,
            150,
        )

        b = rng.randint(
            -50,
            150,
        )

        while b == a:
            b = rng.randint(
                -50,
                150,
            )

        add_objective(
            cases,
            language,
            "comparison",
            compare_template.format(
                a=a,
                b=b,
            ),
            max(a, b),
            evaluator="numeric",
        )

    # ========================================================
    # 3. STRING MANIPULATION - 10
    # ========================================================

    strings = {
        "en": [
            "NeuralNetwork",
            "OpenScience",
            "DataModel",
            "GreenForest",
            "VectorSpace",
        ],

        "pt": [
            "RedeNeural",
            "CienciaAberta",
            "ModeloDados",
            "FlorestaVerde",
            "EspacoVetorial",
        ],

        "es": [
            "RedNeuronal",
            "CienciaAbierta",
            "ModeloDatos",
            "BosqueVerde",
            "EspacioVectorial",
        ],
    }[language]

    for text in strings:
        add_objective(
            cases,
            language,
            "string",
            lowercase_template.format(
                text=text
            ),
            text.lower(),
        )

    for text in strings:
        add_objective(
            cases,
            language,
            "string",
            uppercase_template.format(
                text=text
            ),
            text.upper(),
        )

    # ========================================================
    # 4. LOGIC / SIMPLE SYMBOLIC REASONING - 10
    # ========================================================

    if language == "en":
        logic_cases = [
            (
                "If all tulips are flowers and all flowers "
                "are plants, are all tulips plants? "
                "Answer yes or no.",
                "yes",
                ["Yes", "YES"],
            ),
            (
                "If no square is a circle and object A is "
                "a square, can object A be a circle? "
                "Answer yes or no.",
                "no",
                ["No", "NO"],
            ),
            (
                "A switch is ON. It is toggled twice. "
                "Is it ON or OFF?",
                "ON",
                ["on"],
            ),
            (
                "A switch is OFF. It is toggled three times. "
                "Is it ON or OFF?",
                "ON",
                ["on"],
            ),
            (
                "If P is true and Q is false, what is "
                "P AND Q? Answer true or false.",
                "false",
                ["False", "FALSE"],
            ),
            (
                "If P is true and Q is false, what is "
                "P OR Q? Answer true or false.",
                "true",
                ["True", "TRUE"],
            ),
            (
                "Sequence: 2, 4, 6, 8. "
                "What is the next number? "
                "Answer with only the number.",
                "10",
                [],
            ),
            (
                "Sequence: 3, 6, 12, 24. "
                "What is the next number? "
                "Answer with only the number.",
                "48",
                [],
            ),
            (
                "Ana is older than Bruno. Bruno is older "
                "than Carla. Who is oldest? "
                "Answer only the name.",
                "Ana",
                [],
            ),
            (
                "Box A is inside Box B. Box B is inside "
                "Box C. Is Box A inside Box C? "
                "Answer yes or no.",
                "yes",
                ["Yes", "YES"],
            ),
        ]

    elif language == "pt":
        logic_cases = [
            (
                "Se todas as tulipas são flores e todas "
                "as flores são plantas, todas as tulipas "
                "são plantas? Responda sim ou não.",
                "sim",
                ["Sim", "SIM"],
            ),
            (
                "Se nenhum quadrado é um círculo e o objeto "
                "A é um quadrado, o objeto A pode ser um "
                "círculo? Responda sim ou não.",
                "não",
                ["nao", "Não", "Nao"],
            ),
            (
                "Um interruptor está LIGADO. Ele é alternado "
                "duas vezes. Ele termina LIGADO ou DESLIGADO?",
                "LIGADO",
                ["ligado"],
            ),
            (
                "Um interruptor está DESLIGADO. Ele é "
                "alternado três vezes. Ele termina LIGADO "
                "ou DESLIGADO?",
                "LIGADO",
                ["ligado"],
            ),
            (
                "Se P é verdadeiro e Q é falso, qual é o "
                "resultado de P E Q? Responda verdadeiro "
                "ou falso.",
                "falso",
                ["Falso", "FALSO"],
            ),
            (
                "Se P é verdadeiro e Q é falso, qual é o "
                "resultado de P OU Q? Responda verdadeiro "
                "ou falso.",
                "verdadeiro",
                ["Verdadeiro", "VERDADEIRO"],
            ),
            (
                "Sequência: 2, 4, 6, 8. "
                "Qual é o próximo número? "
                "Responda somente com o número.",
                "10",
                [],
            ),
            (
                "Sequência: 3, 6, 12, 24. "
                "Qual é o próximo número? "
                "Responda somente com o número.",
                "48",
                [],
            ),
            (
                "Ana é mais velha que Bruno. Bruno é mais "
                "velho que Carla. Quem é a pessoa mais "
                "velha? Responda somente com o nome.",
                "Ana",
                [],
            ),
            (
                "A Caixa A está dentro da Caixa B. A Caixa B "
                "está dentro da Caixa C. A Caixa A está "
                "dentro da Caixa C? Responda sim ou não.",
                "sim",
                ["Sim", "SIM"],
            ),
        ]

    else:
        logic_cases = [
            (
                "Si todos los tulipanes son flores y todas "
                "las flores son plantas, ¿todos los "
                "tulipanes son plantas? Responde sí o no.",
                "sí",
                ["si", "Sí", "Si"],
            ),
            (
                "Si ningún cuadrado es un círculo y el objeto "
                "A es un cuadrado, ¿puede el objeto A ser "
                "un círculo? Responde sí o no.",
                "no",
                ["No", "NO"],
            ),
            (
                "Un interruptor está ENCENDIDO. Se cambia "
                "dos veces. ¿Termina ENCENDIDO o APAGADO?",
                "ENCENDIDO",
                ["encendido"],
            ),
            (
                "Un interruptor está APAGADO. Se cambia tres "
                "veces. ¿Termina ENCENDIDO o APAGADO?",
                "ENCENDIDO",
                ["encendido"],
            ),
            (
                "Si P es verdadero y Q es falso, ¿cuál es "
                "el resultado de P Y Q? Responde verdadero "
                "o falso.",
                "falso",
                ["Falso", "FALSO"],
            ),
            (
                "Si P es verdadero y Q es falso, ¿cuál es "
                "el resultado de P O Q? Responde verdadero "
                "o falso.",
                "verdadero",
                ["Verdadero", "VERDADERO"],
            ),
            (
                "Secuencia: 2, 4, 6, 8. "
                "¿Cuál es el siguiente número? "
                "Responde solamente con el número.",
                "10",
                [],
            ),
            (
                "Secuencia: 3, 6, 12, 24. "
                "¿Cuál es el siguiente número? "
                "Responde solamente con el número.",
                "48",
                [],
            ),
            (
                "Ana es mayor que Bruno. Bruno es mayor "
                "que Carla. ¿Quién es la persona mayor? "
                "Responde solamente con el nombre.",
                "Ana",
                [],
            ),
            (
                "La Caja A está dentro de la Caja B. "
                "La Caja B está dentro de la Caja C. "
                "¿La Caja A está dentro de la Caja C? "
                "Responde sí o no.",
                "sí",
                ["si", "Sí", "Si"],
            ),
        ]

    for (
        prompt,
        answer,
        aliases,
    ) in logic_cases:
        add_objective(
            cases,
            language,
            "logic",
            prompt,
            answer,
            aliases=aliases,
        )

    # ========================================================
    # 5. FACTUAL / SCIENTIFIC - 10
    #
    # Stable basic knowledge only.
    # ========================================================

    if language == "en":
        factual_cases = [
            (
                "What gas do plants primarily absorb from "
                "the atmosphere during photosynthesis? "
                "Answer only the gas name.",
                "carbon dioxide",
                ["CO2", "carbon dioxide (CO2)"],
            ),
            (
                "What is the chemical symbol for water? "
                "Answer only the formula.",
                "H2O",
                [],
            ),
            (
                "How many planets are in the Solar System? "
                "Answer only the number.",
                "8",
                [],
            ),
            (
                "Which organ pumps blood through the human "
                "body? Answer only the organ name.",
                "heart",
                [],
            ),
            (
                "What force attracts objects toward Earth? "
                "Answer with one word.",
                "gravity",
                [],
            ),
            (
                "What is the SI unit of electric current? "
                "Answer only the unit name.",
                "ampere",
                ["amp"],
            ),
            (
                "Which particle has a negative electric "
                "charge: proton, neutron, or electron? "
                "Answer only the particle name.",
                "electron",
                [],
            ),
            (
                "At standard atmospheric pressure, water "
                "freezes at how many degrees Celsius? "
                "Answer only the number.",
                "0",
                [],
            ),
            (
                "DNA stands for deoxyribonucleic acid. "
                "Is this statement true or false?",
                "true",
                ["True", "TRUE"],
            ),
            (
                "Which planet is closest to the Sun? "
                "Answer only the planet name.",
                "Mercury",
                ["mercury"],
            ),
        ]

    elif language == "pt":
        factual_cases = [
            (
                "Qual gás as plantas absorvem principalmente "
                "da atmosfera durante a fotossíntese? "
                "Responda somente com o nome do gás.",
                "dióxido de carbono",
                [
                    "CO2",
                    "dioxido de carbono",
                ],
            ),
            (
                "Qual é a fórmula química da água? "
                "Responda somente com a fórmula.",
                "H2O",
                [],
            ),
            (
                "Quantos planetas existem no Sistema Solar? "
                "Responda somente com o número.",
                "8",
                [],
            ),
            (
                "Qual órgão bombeia sangue pelo corpo humano? "
                "Responda somente com o nome do órgão.",
                "coração",
                ["coracao"],
            ),
            (
                "Qual força atrai os objetos em direção à "
                "Terra? Responda com uma palavra.",
                "gravidade",
                [],
            ),
            (
                "Qual é a unidade SI de corrente elétrica? "
                "Responda somente com o nome da unidade.",
                "ampere",
                ["ampère"],
            ),
            (
                "Qual partícula possui carga elétrica "
                "negativa: próton, nêutron ou elétron? "
                "Responda somente com o nome.",
                "elétron",
                ["eletron"],
            ),
            (
                "À pressão atmosférica padrão, a água "
                "congela a quantos graus Celsius? "
                "Responda somente com o número.",
                "0",
                [],
            ),
            (
                "DNA significa ácido desoxirribonucleico. "
                "Essa afirmação é verdadeira ou falsa?",
                "verdadeira",
                [
                    "verdadeiro",
                    "Verdadeira",
                ],
            ),
            (
                "Qual planeta está mais próximo do Sol? "
                "Responda somente com o nome do planeta.",
                "Mercúrio",
                ["Mercurio"],
            ),
        ]

    else:
        factual_cases = [
            (
                "¿Qué gas absorben principalmente las plantas "
                "de la atmósfera durante la fotosíntesis? "
                "Responde solamente con el nombre del gas.",
                "dióxido de carbono",
                [
                    "CO2",
                    "dioxido de carbono",
                ],
            ),
            (
                "¿Cuál es la fórmula química del agua? "
                "Responde solamente con la fórmula.",
                "H2O",
                [],
            ),
            (
                "¿Cuántos planetas hay en el Sistema Solar? "
                "Responde solamente con el número.",
                "8",
                [],
            ),
            (
                "¿Qué órgano bombea sangre por el cuerpo "
                "humano? Responde solamente con el nombre.",
                "corazón",
                ["corazon"],
            ),
            (
                "¿Qué fuerza atrae los objetos hacia la "
                "Tierra? Responde con una palabra.",
                "gravedad",
                [],
            ),
            (
                "¿Cuál es la unidad SI de corriente eléctrica? "
                "Responde solamente con el nombre.",
                "amperio",
                ["ampere"],
            ),
            (
                "¿Qué partícula tiene carga eléctrica "
                "negativa: protón, neutrón o electrón? "
                "Responde solamente con el nombre.",
                "electrón",
                ["electron"],
            ),
            (
                "A presión atmosférica estándar, ¿a cuántos "
                "grados Celsius se congela el agua? "
                "Responde solamente con el número.",
                "0",
                [],
            ),
            (
                "DNA significa ácido desoxirribonucleico. "
                "¿Esta afirmación es verdadera o falsa?",
                "verdadera",
                [
                    "verdadero",
                    "Verdadera",
                ],
            ),
            (
                "¿Qué planeta está más cerca del Sol? "
                "Responde solamente con el nombre.",
                "Mercurio",
                ["mercurio"],
            ),
        ]

    for (
        prompt,
        answer,
        aliases,
    ) in factual_cases:
        add_objective(
            cases,
            language,
            "science",
            prompt,
            answer,
            aliases=aliases,
        )

    # ========================================================
    # 6. CODE - 5
    # ========================================================

    if language == "en":
        code_cases = [
            (
                "In Python, what is the value of "
                "`len([10, 20, 30])`? "
                "Answer only the number.",
                "3",
            ),
            (
                "In Python, what is the value of "
                "`2 ** 5`? Answer only the number.",
                "32",
            ),
            (
                "In Python, what boolean value is produced by "
                "`5 > 9`? Answer only True or False.",
                "False",
            ),
            (
                "In Python, what is the result of "
                "`'maia'.upper()`? "
                "Answer only the resulting string.",
                "MAIA",
            ),
            (
                "In Python, what is the value of "
                "`sum([1, 2, 3, 4])`? "
                "Answer only the number.",
                "10",
            ),
        ]

    elif language == "pt":
        code_cases = [
            (
                "Em Python, qual é o valor de "
                "`len([10, 20, 30])`? "
                "Responda somente com o número.",
                "3",
            ),
            (
                "Em Python, qual é o valor de "
                "`2 ** 5`? "
                "Responda somente com o número.",
                "32",
            ),
            (
                "Em Python, qual valor booleano é produzido "
                "por `5 > 9`? "
                "Responda somente True ou False.",
                "False",
            ),
            (
                "Em Python, qual é o resultado de "
                "`'maia'.upper()`? "
                "Responda somente com a string resultante.",
                "MAIA",
            ),
            (
                "Em Python, qual é o valor de "
                "`sum([1, 2, 3, 4])`? "
                "Responda somente com o número.",
                "10",
            ),
        ]

    else:
        code_cases = [
            (
                "En Python, ¿cuál es el valor de "
                "`len([10, 20, 30])`? "
                "Responde solamente con el número.",
                "3",
            ),
            (
                "En Python, ¿cuál es el valor de "
                "`2 ** 5`? "
                "Responde solamente con el número.",
                "32",
            ),
            (
                "En Python, ¿qué valor booleano produce "
                "`5 > 9`? "
                "Responde solamente True o False.",
                "False",
            ),
            (
                "En Python, ¿cuál es el resultado de "
                "`'maia'.upper()`? "
                "Responde solamente con la cadena resultante.",
                "MAIA",
            ),
            (
                "En Python, ¿cuál es el valor de "
                "`sum([1, 2, 3, 4])`? "
                "Responde solamente con el número.",
                "10",
            ),
        ]

    for prompt, answer in code_cases:
        add_objective(
            cases,
            language,
            "code",
            prompt,
            answer,
        )

    # ========================================================
    # EXACTLY 60 OBJECTIVE PER LANGUAGE
    # ========================================================

    objective_count = sum(
        1
        for case in cases
        if (
            case["language"]
            == language
            and case["kind"]
            == "objective"
        )
    )

    if objective_count != 60:
        raise RuntimeError(
            f"{language}: expected "
            f"60 objective cases, "
            f"got {objective_count}"
        )


# ============================================================
# QUALITATIVE BENCHMARK
# ============================================================


def qualitative_templates(
    language,
):
    if language == "en":
        return {
            "explanation": [
                (
                    "Explain why the sky appears blue during "
                    "the day to a curious 12-year-old.",
                    [
                        "answers the question",
                        "uses clear age-appropriate language",
                        "avoids unnecessary jargon",
                        "does not introduce major factual errors",
                    ],
                ),
                (
                    "Explain the difference between RAM and "
                    "persistent storage in a computer.",
                    [
                        "distinguishes volatile and persistent storage",
                        "explains practical roles",
                        "is concise and coherent",
                    ],
                ),
                (
                    "Explain what overfitting means in machine "
                    "learning using a simple analogy.",
                    [
                        "captures poor generalization",
                        "uses a useful analogy",
                        "remains technically consistent",
                    ],
                ),
                (
                    "Explain why scientific experiments use "
                    "control groups.",
                    [
                        "explains comparison/baseline role",
                        "connects controls to causal interpretation",
                        "is understandable",
                    ],
                ),
                (
                    "Explain recursion in programming without "
                    "using mathematical notation.",
                    [
                        "describes self-reference or repeated subproblems",
                        "mentions a stopping/base condition",
                        "uses accessible language",
                    ],
                ),
            ],

            "summarization": [
                (
                    "Summarize this text in one sentence: "
                    "\"Solar panels convert sunlight into "
                    "electricity. Their output depends on "
                    "illumination, orientation, temperature, "
                    "and system efficiency.\"",
                    [
                        "preserves the central idea",
                        "mentions relevant output factors",
                        "uses one sentence",
                    ],
                ),
                (
                    "Summarize in two short sentences: "
                    "\"Version control records changes to files "
                    "over time. It allows developers to compare "
                    "versions, recover earlier states, and "
                    "collaborate safely.\"",
                    [
                        "captures version history",
                        "captures recovery/collaboration",
                        "uses no more than two sentences",
                    ],
                ),
                (
                    "Summarize in one sentence: "
                    "\"Vaccination exposes the immune system to "
                    "a safe representation of a pathogen or its "
                    "components, helping immune memory develop.\"",
                    [
                        "preserves immune exposure concept",
                        "preserves immune memory concept",
                        "does not claim vaccination causes disease",
                    ],
                ),
                (
                    "Summarize in one sentence: "
                    "\"A database index uses additional data "
                    "structures to accelerate searches, usually "
                    "at the cost of extra storage and update work.\"",
                    [
                        "mentions faster lookup",
                        "mentions tradeoff",
                        "uses one sentence",
                    ],
                ),
            ],

            "rewriting": [
                (
                    "Rewrite this message in a professional and "
                    "friendly tone: \"send me the report now "
                    "because I need it\".",
                    [
                        "preserves the request",
                        "improves politeness",
                        "does not invent unrelated details",
                    ],
                ),
                (
                    "Rewrite for clarity: \"The program doesn't "
                    "work sometimes and it gives some error when "
                    "the file is big.\"",
                    [
                        "preserves intermittent failure",
                        "preserves relation to large files",
                        "is clearer than source",
                    ],
                ),
                (
                    "Rewrite concisely: \"Due to the fact that "
                    "the server was unavailable, we were unable "
                    "to complete the deployment at that point "
                    "in time.\"",
                    [
                        "preserves cause",
                        "preserves failed deployment",
                        "is substantially more concise",
                    ],
                ),
                (
                    "Rewrite this as a neutral technical statement: "
                    "\"This terrible algorithm wastes a ridiculous "
                    "amount of memory.\"",
                    [
                        "removes emotional language",
                        "preserves memory-efficiency criticism",
                        "sounds technical",
                    ],
                ),
            ],

            "translation": [
                (
                    "Translate to Portuguese: "
                    "\"The model completed training successfully.\"",
                    [
                        "correct Portuguese translation",
                        "preserves meaning",
                        "does not add information",
                    ],
                ),
                (
                    "Translate to Spanish: "
                    "\"The dataset contains scientific articles.\"",
                    [
                        "correct Spanish translation",
                        "preserves meaning",
                        "does not add information",
                    ],
                ),
                (
                    "Translate to English: "
                    "\"A pesquisa precisa de mais dados.\"",
                    [
                        "correct English translation",
                        "preserves meaning",
                    ],
                ),
                (
                    "Translate to English: "
                    "\"El sistema puede ejecutarse sin conexión.\"",
                    [
                        "correct English translation",
                        "preserves offline meaning",
                    ],
                ),
            ],

            "coding": [
                (
                    "Write a Python function `is_even(n)` that "
                    "returns True when n is even and False otherwise.",
                    [
                        "valid Python",
                        "correct parity behavior",
                        "defines requested function",
                    ],
                ),
                (
                    "Write a Python function that returns the "
                    "largest value in a non-empty list without "
                    "sorting the list.",
                    [
                        "valid Python",
                        "does not require sorting",
                        "returns correct maximum",
                    ],
                ),
                (
                    "Explain what is wrong with this Python code "
                    "and provide a correction: `if x = 5: print(x)`",
                    [
                        "identifies assignment/comparison issue",
                        "uses == in corrected condition",
                        "provides valid correction",
                    ],
                ),
                (
                    "Write a short Python example that reads a "
                    "text file using a context manager.",
                    [
                        "uses with/open",
                        "reads the file",
                        "is syntactically plausible Python",
                    ],
                ),
                (
                    "Write a Python function that counts how many "
                    "times a given value appears in a list.",
                    [
                        "defines a function",
                        "accepts list/value or equivalent",
                        "produces correct count",
                    ],
                ),
            ],

            "science_reasoning": [
                (
                    "A plant kept in darkness for several days "
                    "receives water but no light. Explain why its "
                    "ability to produce new sugars is reduced.",
                    [
                        "connects light to photosynthesis",
                        "connects photosynthesis to sugar production",
                        "does not claim water alone replaces light",
                    ],
                ),
                (
                    "Two identical metal objects have different "
                    "temperatures and are placed in contact. "
                    "Explain the expected direction of heat flow.",
                    [
                        "heat flows hot to cold",
                        "mentions approach toward thermal equilibrium",
                        "does not reverse direction",
                    ],
                ),
                (
                    "Why does increasing a sample size often make "
                    "an estimated mean more stable?",
                    [
                        "connects larger samples to reduced sampling variability",
                        "does not claim all bias disappears",
                        "provides coherent explanation",
                    ],
                ),
                (
                    "A program becomes slower as its input grows. "
                    "Explain why measuring runtime at several input "
                    "sizes is more informative than measuring it once.",
                    [
                        "mentions scaling behavior",
                        "distinguishes size-dependent performance",
                        "supports multiple measurements",
                    ],
                ),
                (
                    "Explain why correlation alone does not establish "
                    "that one variable causes another.",
                    [
                        "distinguishes association from causation",
                        "mentions confounding or alternative explanations",
                        "avoids claiming correlation is useless",
                    ],
                ),
            ],

            "uncertainty": [
                (
                    "A user asks: \"What will the exact temperature "
                    "in my city be exactly one year from today?\" "
                    "Answer helpfully without pretending to know.",
                    [
                        "acknowledges uncertainty",
                        "does not invent an exact forecast",
                        "offers a useful alternative",
                    ],
                ),
                (
                    "A user provides only the sentence \"it stopped "
                    "working\" and asks for the exact software bug. "
                    "Respond appropriately.",
                    [
                        "does not invent a specific bug",
                        "requests relevant diagnostic information",
                        "remains helpful",
                    ],
                ),
                (
                    "Someone asks whether an unknown experimental "
                    "drug will definitely cure a disease. Respond "
                    "carefully.",
                    [
                        "does not guarantee effectiveness",
                        "acknowledges evidence/uncertainty",
                        "uses appropriately cautious language",
                    ],
                ),
                (
                    "A user asks for the contents of a document that "
                    "has not been provided. Respond appropriately.",
                    [
                        "does not invent document contents",
                        "states that the document is needed",
                        "gives a useful next step",
                    ],
                ),
            ],

            "formatting": [
                (
                    "Give exactly three bullet points describing "
                    "benefits of automated testing.",
                    [
                        "exactly three bullet items",
                        "items concern automated testing",
                        "clear formatting",
                    ],
                ),
                (
                    "Return a two-column Markdown table with headers "
                    "`Term` and `Meaning` for CPU, RAM, and SSD.",
                    [
                        "Markdown table structure",
                        "correct headers",
                        "contains CPU RAM SSD",
                    ],
                ),
                (
                    "Describe the software workflow Plan → Implement "
                    "→ Test → Deploy as exactly four numbered steps.",
                    [
                        "exactly four numbered steps",
                        "preserves requested order",
                        "uses requested concepts",
                    ],
                ),
                (
                    "Give a JSON object with keys `language` and "
                    "`purpose`, using values `Python` and "
                    "`programming` respectively. Return only JSON.",
                    [
                        "valid JSON object",
                        "contains exact requested keys",
                        "contains requested values",
                    ],
                ),
            ],

            "multi_turn": [
                (
                    [
                        message(
                            "user",
                            "I am planning a small experiment. "
                            "Remember that my sample size is 40.",
                        ),
                        message(
                            "assistant",
                            "Understood. Your sample size is 40.",
                        ),
                        message(
                            "user",
                            "If I divide the sample equally into "
                            "four groups, how many observations "
                            "are in each group?",
                        ),
                    ],
                    [
                        "uses prior-turn sample size",
                        "answers 10",
                        "does not lose conversational context",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "For this conversation, call the first "
                            "dataset Alpha and the second Beta.",
                        ),
                        message(
                            "assistant",
                            "Understood.",
                        ),
                        message(
                            "user",
                            "Which name refers to the second dataset?",
                        ),
                    ],
                    [
                        "answers Beta",
                        "uses prior-turn naming",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "My preferred output format for this task "
                            "is a numbered list.",
                        ),
                        message(
                            "assistant",
                            "Understood.",
                        ),
                        message(
                            "user",
                            "Give two steps for saving a text file.",
                        ),
                    ],
                    [
                        "uses numbered list",
                        "contains two steps",
                        "answers requested task",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "Suppose x equals 12.",
                        ),
                        message(
                            "assistant",
                            "Okay, x equals 12.",
                        ),
                        message(
                            "user",
                            "What is x multiplied by 3?",
                        ),
                    ],
                    [
                        "uses x=12 from context",
                        "answers 36",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "We are discussing photosynthesis.",
                        ),
                        message(
                            "assistant",
                            "Understood.",
                        ),
                        message(
                            "user",
                            "In one sentence, state the role of light "
                            "in the process we are discussing.",
                        ),
                    ],
                    [
                        "maintains photosynthesis context",
                        "connects light to energy/photosynthesis",
                        "uses one sentence",
                    ],
                ),
            ],
        }

    if language == "pt":
        return {
            "explanation": [
                (
                    "Explique por que o céu parece azul durante "
                    "o dia para uma criança curiosa de 12 anos.",
                    [
                        "responde à pergunta",
                        "usa linguagem clara e apropriada",
                        "evita erros factuais importantes",
                    ],
                ),
                (
                    "Explique a diferença entre memória RAM e "
                    "armazenamento persistente em um computador.",
                    [
                        "distingue memória volátil de persistente",
                        "explica os papéis práticos",
                        "é coerente",
                    ],
                ),
                (
                    "Explique o que significa overfitting em "
                    "aprendizado de máquina usando uma analogia simples.",
                    [
                        "explica generalização ruim",
                        "usa analogia útil",
                        "mantém consistência técnica",
                    ],
                ),
                (
                    "Explique por que experimentos científicos "
                    "utilizam grupos de controle.",
                    [
                        "explica o papel de comparação",
                        "relaciona controle à interpretação causal",
                        "é compreensível",
                    ],
                ),
                (
                    "Explique recursão em programação sem usar "
                    "notação matemática.",
                    [
                        "explica autorreferência ou subproblemas",
                        "menciona condição de parada",
                        "usa linguagem acessível",
                    ],
                ),
            ],

            "summarization": [
                (
                    "Resuma em uma frase: \"Painéis solares "
                    "convertem luz solar em eletricidade. Sua "
                    "produção depende da iluminação, orientação, "
                    "temperatura e eficiência do sistema.\"",
                    [
                        "preserva a ideia central",
                        "menciona fatores relevantes",
                        "usa uma frase",
                    ],
                ),
                (
                    "Resuma em duas frases curtas: \"O controle "
                    "de versão registra alterações em arquivos ao "
                    "longo do tempo. Ele permite comparar versões, "
                    "recuperar estados anteriores e colaborar com "
                    "segurança.\"",
                    [
                        "captura histórico de versões",
                        "captura recuperação ou colaboração",
                        "usa no máximo duas frases",
                    ],
                ),
                (
                    "Resuma em uma frase: \"A vacinação expõe o "
                    "sistema imunológico a uma representação segura "
                    "de um patógeno ou de seus componentes, ajudando "
                    "a desenvolver memória imunológica.\"",
                    [
                        "preserva exposição imunológica",
                        "preserva memória imunológica",
                        "não afirma que vacinação causa a doença",
                    ],
                ),
                (
                    "Resuma em uma frase: \"Um índice de banco de "
                    "dados utiliza estruturas adicionais para "
                    "acelerar buscas, geralmente ao custo de mais "
                    "armazenamento e trabalho nas atualizações.\"",
                    [
                        "menciona busca mais rápida",
                        "menciona tradeoff",
                        "usa uma frase",
                    ],
                ),
            ],

            "rewriting": [
                (
                    "Reescreva de forma profissional e cordial: "
                    "\"mande o relatório agora porque eu preciso dele\".",
                    [
                        "preserva o pedido",
                        "melhora a cordialidade",
                        "não inventa detalhes",
                    ],
                ),
                (
                    "Reescreva com maior clareza: \"O programa às "
                    "vezes não funciona e dá algum erro quando o "
                    "arquivo é grande.\"",
                    [
                        "preserva falha intermitente",
                        "preserva relação com arquivos grandes",
                        "melhora clareza",
                    ],
                ),
                (
                    "Reescreva de forma concisa: \"Devido ao fato "
                    "de o servidor estar indisponível, não foi "
                    "possível concluir a implantação naquele "
                    "momento.\"",
                    [
                        "preserva a causa",
                        "preserva a implantação não concluída",
                        "é mais conciso",
                    ],
                ),
                (
                    "Reescreva como uma afirmação técnica neutra: "
                    "\"Esse algoritmo horrível desperdiça uma "
                    "quantidade ridícula de memória.\"",
                    [
                        "remove linguagem emocional",
                        "preserva crítica de eficiência de memória",
                        "usa tom técnico",
                    ],
                ),
            ],

            "translation": [
                (
                    "Traduza para inglês: "
                    "\"O modelo concluiu o treinamento com sucesso.\"",
                    [
                        "tradução correta para inglês",
                        "preserva significado",
                    ],
                ),
                (
                    "Traduza para espanhol: "
                    "\"O conjunto de dados contém artigos científicos.\"",
                    [
                        "tradução correta para espanhol",
                        "preserva significado",
                    ],
                ),
                (
                    "Traduza para português: "
                    "\"The system can operate offline.\"",
                    [
                        "tradução correta para português",
                        "preserva significado de offline",
                    ],
                ),
                (
                    "Traduza para português: "
                    "\"La investigación necesita más datos.\"",
                    [
                        "tradução correta para português",
                        "preserva significado",
                    ],
                ),
            ],

            "coding": [
                (
                    "Escreva uma função Python `is_even(n)` que "
                    "retorne True quando n for par e False caso contrário.",
                    [
                        "Python válido",
                        "comportamento correto para paridade",
                        "define a função solicitada",
                    ],
                ),
                (
                    "Escreva uma função Python que retorne o maior "
                    "valor de uma lista não vazia sem ordenar a lista.",
                    [
                        "Python válido",
                        "não exige ordenação",
                        "retorna máximo correto",
                    ],
                ),
                (
                    "Explique o erro neste código Python e apresente "
                    "a correção: `if x = 5: print(x)`",
                    [
                        "identifica atribuição versus comparação",
                        "usa == na correção",
                        "fornece código válido",
                    ],
                ),
                (
                    "Escreva um pequeno exemplo em Python que leia "
                    "um arquivo de texto usando um gerenciador de contexto.",
                    [
                        "usa with/open",
                        "lê o arquivo",
                        "é Python sintaticamente plausível",
                    ],
                ),
                (
                    "Escreva uma função Python que conte quantas "
                    "vezes um determinado valor aparece em uma lista.",
                    [
                        "define uma função",
                        "recebe lista e valor ou equivalente",
                        "produz contagem correta",
                    ],
                ),
            ],

            "science_reasoning": [
                (
                    "Uma planta mantida no escuro por vários dias "
                    "recebe água, mas nenhuma luz. Explique por que "
                    "sua capacidade de produzir novos açúcares diminui.",
                    [
                        "relaciona luz à fotossíntese",
                        "relaciona fotossíntese à produção de açúcar",
                        "não afirma que água substitui luz",
                    ],
                ),
                (
                    "Dois objetos metálicos idênticos estão em "
                    "temperaturas diferentes e são colocados em "
                    "contato. Explique a direção esperada do fluxo "
                    "de calor.",
                    [
                        "calor flui do quente para o frio",
                        "menciona equilíbrio térmico",
                        "não inverte a direção",
                    ],
                ),
                (
                    "Por que aumentar o tamanho de uma amostra "
                    "frequentemente torna uma média estimada mais estável?",
                    [
                        "relaciona amostra maior a menor variabilidade",
                        "não afirma que todo viés desaparece",
                        "explicação coerente",
                    ],
                ),
                (
                    "Um programa fica mais lento conforme sua entrada "
                    "cresce. Explique por que medir o tempo de execução "
                    "para vários tamanhos de entrada é mais informativo "
                    "do que medir apenas uma vez.",
                    [
                        "menciona comportamento de escala",
                        "relaciona desempenho ao tamanho",
                        "justifica múltiplas medições",
                    ],
                ),
                (
                    "Explique por que correlação, isoladamente, não "
                    "estabelece que uma variável causa outra.",
                    [
                        "distingue associação de causalidade",
                        "menciona confundimento ou explicações alternativas",
                        "não diz que correlação é inútil",
                    ],
                ),
            ],

            "uncertainty": [
                (
                    "Um usuário pergunta: \"Qual será exatamente a "
                    "temperatura na minha cidade daqui a um ano?\" "
                    "Responda de forma útil sem fingir que sabe.",
                    [
                        "reconhece incerteza",
                        "não inventa previsão exata",
                        "oferece alternativa útil",
                    ],
                ),
                (
                    "Um usuário fornece apenas a frase \"parou de "
                    "funcionar\" e pede para identificar exatamente "
                    "o bug do software. Responda adequadamente.",
                    [
                        "não inventa bug específico",
                        "pede informações diagnósticas",
                        "permanece útil",
                    ],
                ),
                (
                    "Alguém pergunta se um medicamento experimental "
                    "desconhecido certamente curará uma doença. "
                    "Responda com cautela.",
                    [
                        "não garante eficácia",
                        "reconhece necessidade de evidência",
                        "usa linguagem cautelosa",
                    ],
                ),
                (
                    "Um usuário pede o conteúdo de um documento que "
                    "não foi fornecido. Responda adequadamente.",
                    [
                        "não inventa o conteúdo",
                        "informa que precisa do documento",
                        "indica próximo passo",
                    ],
                ),
            ],

            "formatting": [
                (
                    "Forneça exatamente três itens com marcadores "
                    "descrevendo benefícios de testes automatizados.",
                    [
                        "exatamente três itens",
                        "itens tratam de testes automatizados",
                        "formatação clara",
                    ],
                ),
                (
                    "Retorne uma tabela Markdown de duas colunas com "
                    "cabeçalhos `Termo` e `Significado` para CPU, "
                    "RAM e SSD.",
                    [
                        "estrutura de tabela Markdown",
                        "cabeçalhos corretos",
                        "contém CPU RAM SSD",
                    ],
                ),
                (
                    "Descreva o fluxo Planejar → Implementar → Testar "
                    "→ Implantar como exatamente quatro passos numerados.",
                    [
                        "exatamente quatro passos",
                        "preserva a ordem",
                        "usa os conceitos solicitados",
                    ],
                ),
                (
                    "Forneça um objeto JSON com as chaves `language` "
                    "e `purpose`, usando respectivamente os valores "
                    "`Python` e `programming`. Retorne somente JSON.",
                    [
                        "objeto JSON válido",
                        "chaves exatas",
                        "valores solicitados",
                    ],
                ),
            ],

            "multi_turn": [
                (
                    [
                        message(
                            "user",
                            "Estou planejando um pequeno experimento. "
                            "Lembre que minha amostra tem tamanho 40.",
                        ),
                        message(
                            "assistant",
                            "Entendido. Sua amostra tem tamanho 40.",
                        ),
                        message(
                            "user",
                            "Se eu dividir a amostra igualmente em "
                            "quatro grupos, quantas observações haverá "
                            "em cada grupo?",
                        ),
                    ],
                    [
                        "usa tamanho 40 do turno anterior",
                        "responde 10",
                        "mantém contexto conversacional",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "Nesta conversa, chame o primeiro conjunto "
                            "de dados de Alfa e o segundo de Beta.",
                        ),
                        message(
                            "assistant",
                            "Entendido.",
                        ),
                        message(
                            "user",
                            "Qual nome se refere ao segundo conjunto?",
                        ),
                    ],
                    [
                        "responde Beta",
                        "usa a nomeação do turno anterior",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "Para esta tarefa, prefiro a saída em "
                            "uma lista numerada.",
                        ),
                        message(
                            "assistant",
                            "Entendido.",
                        ),
                        message(
                            "user",
                            "Forneça dois passos para salvar um "
                            "arquivo de texto.",
                        ),
                    ],
                    [
                        "usa lista numerada",
                        "contém dois passos",
                        "responde à tarefa",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "Suponha que x seja igual a 12.",
                        ),
                        message(
                            "assistant",
                            "Certo, x é igual a 12.",
                        ),
                        message(
                            "user",
                            "Quanto é x multiplicado por 3?",
                        ),
                    ],
                    [
                        "usa x=12 do contexto",
                        "responde 36",
                    ],
                ),
                (
                    [
                        message(
                            "user",
                            "Estamos discutindo fotossíntese.",
                        ),
                        message(
                            "assistant",
                            "Entendido.",
                        ),
                        message(
                            "user",
                            "Em uma frase, indique o papel da luz "
                            "no processo que estamos discutindo.",
                        ),
                    ],
                    [
                        "mantém contexto de fotossíntese",
                        "relaciona luz à energia/fotossíntese",
                        "usa uma frase",
                    ],
                ),
            ],
        }

    # Spanish
    return {
        "explanation": [
            (
                "Explica por qué el cielo parece azul durante el "
                "día a un niño curioso de 12 años.",
                [
                    "responde la pregunta",
                    "usa lenguaje claro",
                    "evita errores factuales importantes",
                ],
            ),
            (
                "Explica la diferencia entre memoria RAM y "
                "almacenamiento persistente en una computadora.",
                [
                    "distingue memoria volátil y persistente",
                    "explica sus funciones",
                    "es coherente",
                ],
            ),
            (
                "Explica qué significa overfitting en aprendizaje "
                "automático usando una analogía sencilla.",
                [
                    "explica mala generalización",
                    "usa una analogía útil",
                    "mantiene consistencia técnica",
                ],
            ),
            (
                "Explica por qué los experimentos científicos "
                "utilizan grupos de control.",
                [
                    "explica función de comparación",
                    "relaciona controles con interpretación causal",
                    "es comprensible",
                ],
            ),
            (
                "Explica la recursión en programación sin utilizar "
                "notación matemática.",
                [
                    "explica autorreferencia o subproblemas",
                    "menciona condición de parada",
                    "usa lenguaje accesible",
                ],
            ),
        ],

        "summarization": [
            (
                "Resume en una oración: \"Los paneles solares "
                "convierten la luz solar en electricidad. Su "
                "producción depende de la iluminación, orientación, "
                "temperatura y eficiencia del sistema.\"",
                [
                    "preserva idea central",
                    "menciona factores relevantes",
                    "usa una oración",
                ],
            ),
            (
                "Resume en dos oraciones cortas: \"El control de "
                "versiones registra cambios en archivos a lo largo "
                "del tiempo. Permite comparar versiones, recuperar "
                "estados anteriores y colaborar de forma segura.\"",
                [
                    "captura historial de versiones",
                    "captura recuperación o colaboración",
                    "usa máximo dos oraciones",
                ],
            ),
            (
                "Resume en una oración: \"La vacunación expone al "
                "sistema inmunitario a una representación segura de "
                "un patógeno o de sus componentes, ayudando a "
                "desarrollar memoria inmunitaria.\"",
                [
                    "preserva exposición inmunitaria",
                    "preserva memoria inmunitaria",
                    "no afirma que vacunación cause enfermedad",
                ],
            ),
            (
                "Resume en una oración: \"Un índice de base de datos "
                "utiliza estructuras adicionales para acelerar las "
                "búsquedas, normalmente a costa de almacenamiento "
                "adicional y trabajo durante las actualizaciones.\"",
                [
                    "menciona búsquedas más rápidas",
                    "menciona tradeoff",
                    "usa una oración",
                ],
            ),
        ],

        "rewriting": [
            (
                "Reescribe de forma profesional y cordial: "
                "\"mándame el informe ahora porque lo necesito\".",
                [
                    "preserva la solicitud",
                    "mejora cortesía",
                    "no inventa detalles",
                ],
            ),
            (
                "Reescribe con mayor claridad: \"El programa a "
                "veces no funciona y da algún error cuando el "
                "archivo es grande.\"",
                [
                    "preserva fallo intermitente",
                    "preserva relación con archivos grandes",
                    "mejora claridad",
                ],
            ),
            (
                "Reescribe de forma concisa: \"Debido al hecho de "
                "que el servidor no estaba disponible, no pudimos "
                "completar el despliegue en ese momento.\"",
                [
                    "preserva causa",
                    "preserva despliegue incompleto",
                    "es más conciso",
                ],
            ),
            (
                "Reescribe como una afirmación técnica neutral: "
                "\"Este algoritmo horrible desperdicia una cantidad "
                "ridícula de memoria.\"",
                [
                    "elimina lenguaje emocional",
                    "preserva crítica de eficiencia",
                    "usa tono técnico",
                ],
            ),
        ],

        "translation": [
            (
                "Traduce al inglés: "
                "\"El modelo completó el entrenamiento correctamente.\"",
                [
                    "traducción correcta al inglés",
                    "preserva significado",
                ],
            ),
            (
                "Traduce al portugués: "
                "\"El conjunto de datos contiene artículos científicos.\"",
                [
                    "traducción correcta al portugués",
                    "preserva significado",
                ],
            ),
            (
                "Traduce al español: "
                "\"The system can operate offline.\"",
                [
                    "traducción correcta al español",
                    "preserva significado de offline",
                ],
            ),
            (
                "Traduce al español: "
                "\"A pesquisa precisa de mais dados.\"",
                [
                    "traducción correcta al español",
                    "preserva significado",
                ],
            ),
        ],

        "coding": [
            (
                "Escribe una función Python `is_even(n)` que "
                "devuelva True cuando n sea par y False en caso contrario.",
                [
                    "Python válido",
                    "comportamiento correcto",
                    "define función solicitada",
                ],
            ),
            (
                "Escribe una función Python que devuelva el valor "
                "máximo de una lista no vacía sin ordenar la lista.",
                [
                    "Python válido",
                    "no requiere ordenar",
                    "devuelve máximo correcto",
                ],
            ),
            (
                "Explica el error en este código Python y proporciona "
                "la corrección: `if x = 5: print(x)`",
                [
                    "identifica asignación/comparación",
                    "usa == en corrección",
                    "proporciona código válido",
                ],
            ),
            (
                "Escribe un ejemplo corto en Python que lea un "
                "archivo de texto usando un administrador de contexto.",
                [
                    "usa with/open",
                    "lee el archivo",
                    "Python sintácticamente plausible",
                ],
            ),
            (
                "Escribe una función Python que cuente cuántas veces "
                "aparece un valor determinado en una lista.",
                [
                    "define una función",
                    "acepta lista y valor o equivalente",
                    "produce conteo correcto",
                ],
            ),
        ],

        "science_reasoning": [
            (
                "Una planta mantenida en oscuridad durante varios "
                "días recibe agua pero ninguna luz. Explica por qué "
                "disminuye su capacidad de producir nuevos azúcares.",
                [
                    "relaciona luz con fotosíntesis",
                    "relaciona fotosíntesis con azúcar",
                    "no afirma que agua sustituya luz",
                ],
            ),
            (
                "Dos objetos metálicos idénticos tienen temperaturas "
                "diferentes y se ponen en contacto. Explica la "
                "dirección esperada del flujo de calor.",
                [
                    "calor fluye de caliente a frío",
                    "menciona equilibrio térmico",
                    "no invierte dirección",
                ],
            ),
            (
                "¿Por qué aumentar el tamaño de una muestra suele "
                "hacer más estable una media estimada?",
                [
                    "relaciona muestra mayor con menor variabilidad",
                    "no afirma que todo sesgo desaparece",
                    "explicación coherente",
                ],
            ),
            (
                "Un programa se vuelve más lento a medida que crece "
                "su entrada. Explica por qué medir el tiempo de "
                "ejecución para varios tamaños es más informativo "
                "que medirlo una sola vez.",
                [
                    "menciona comportamiento de escala",
                    "relaciona rendimiento con tamaño",
                    "justifica varias mediciones",
                ],
            ),
            (
                "Explica por qué la correlación por sí sola no "
                "establece que una variable cause otra.",
                [
                    "distingue asociación de causalidad",
                    "menciona confusión o explicaciones alternativas",
                    "no afirma que correlación sea inútil",
                ],
            ),
        ],

        "uncertainty": [
            (
                "Un usuario pregunta: \"¿Cuál será exactamente la "
                "temperatura en mi ciudad dentro de un año?\" "
                "Responde de forma útil sin fingir que lo sabes.",
                [
                    "reconoce incertidumbre",
                    "no inventa pronóstico exacto",
                    "ofrece alternativa útil",
                ],
            ),
            (
                "Un usuario solamente dice \"dejó de funcionar\" y "
                "pide identificar exactamente el error del software. "
                "Responde adecuadamente.",
                [
                    "no inventa error específico",
                    "solicita información diagnóstica",
                    "permanece útil",
                ],
            ),
            (
                "Alguien pregunta si un medicamento experimental "
                "desconocido curará con certeza una enfermedad. "
                "Responde con cautela.",
                [
                    "no garantiza eficacia",
                    "reconoce necesidad de evidencia",
                    "usa lenguaje cauteloso",
                ],
            ),
            (
                "Un usuario pide el contenido de un documento que "
                "no ha proporcionado. Responde adecuadamente.",
                [
                    "no inventa contenido",
                    "indica que necesita el documento",
                    "ofrece siguiente paso",
                ],
            ),
        ],

        "formatting": [
            (
                "Proporciona exactamente tres elementos con viñetas "
                "sobre beneficios de las pruebas automatizadas.",
                [
                    "exactamente tres elementos",
                    "tratan de pruebas automatizadas",
                    "formato claro",
                ],
            ),
            (
                "Devuelve una tabla Markdown de dos columnas con "
                "encabezados `Término` y `Significado` para CPU, "
                "RAM y SSD.",
                [
                    "estructura de tabla Markdown",
                    "encabezados correctos",
                    "contiene CPU RAM SSD",
                ],
            ),
            (
                "Describe el flujo Planificar → Implementar → Probar "
                "→ Desplegar como exactamente cuatro pasos numerados.",
                [
                    "exactamente cuatro pasos",
                    "preserva orden",
                    "usa conceptos solicitados",
                ],
            ),
            (
                "Devuelve un objeto JSON con las claves `language` "
                "y `purpose`, usando respectivamente los valores "
                "`Python` y `programming`. Devuelve solamente JSON.",
                [
                    "objeto JSON válido",
                    "claves exactas",
                    "valores solicitados",
                ],
            ),
        ],

        "multi_turn": [
            (
                [
                    message(
                        "user",
                        "Estoy planificando un pequeño experimento. "
                        "Recuerda que mi muestra tiene tamaño 40.",
                    ),
                    message(
                        "assistant",
                        "Entendido. Tu muestra tiene tamaño 40.",
                    ),
                    message(
                        "user",
                        "Si divido la muestra por igual en cuatro "
                        "grupos, ¿cuántas observaciones habrá en cada uno?",
                    ),
                ],
                [
                    "usa tamaño 40 del turno anterior",
                    "responde 10",
                    "mantiene contexto conversacional",
                ],
            ),
            (
                [
                    message(
                        "user",
                        "En esta conversación, llama Alfa al primer "
                        "conjunto de datos y Beta al segundo.",
                    ),
                    message(
                        "assistant",
                        "Entendido.",
                    ),
                    message(
                        "user",
                        "¿Qué nombre corresponde al segundo conjunto?",
                    ),
                ],
                [
                    "responde Beta",
                    "usa nombres del turno anterior",
                ],
            ),
            (
                [
                    message(
                        "user",
                        "Para esta tarea prefiero la salida como "
                        "una lista numerada.",
                    ),
                    message(
                        "assistant",
                        "Entendido.",
                    ),
                    message(
                        "user",
                        "Proporciona dos pasos para guardar un "
                        "archivo de texto.",
                    ),
                ],
                [
                    "usa lista numerada",
                    "contiene dos pasos",
                    "responde tarea solicitada",
                ],
            ),
            (
                [
                    message(
                        "user",
                        "Supongamos que x es igual a 12.",
                    ),
                    message(
                        "assistant",
                        "De acuerdo, x es igual a 12.",
                    ),
                    message(
                        "user",
                        "¿Cuánto es x multiplicado por 3?",
                    ),
                ],
                [
                    "usa x=12 del contexto",
                    "responde 36",
                ],
            ),
            (
                [
                    message(
                        "user",
                        "Estamos hablando de fotosíntesis.",
                    ),
                    message(
                        "assistant",
                        "Entendido.",
                    ),
                    message(
                        "user",
                        "En una oración, indica el papel de la luz "
                        "en el proceso que estamos discutiendo.",
                    ),
                ],
                [
                    "mantiene contexto de fotosíntesis",
                    "relaciona luz con energía/fotosíntesis",
                    "usa una oración",
                ],
            ),
        ],
    }


def build_qualitative(
    cases,
    language,
):
    groups = qualitative_templates(
        language
    )

    expected_counts = {
        "explanation": 5,
        "summarization": 4,
        "rewriting": 4,
        "translation": 4,
        "coding": 5,
        "science_reasoning": 5,
        "uncertainty": 4,
        "formatting": 4,
        "multi_turn": 5,
    }

    for (
        category,
        expected_count,
    ) in expected_counts.items():
        entries = groups[
            category
        ]

        if len(entries) != expected_count:
            raise RuntimeError(
                f"{language}/{category}: "
                f"expected {expected_count}, "
                f"got {len(entries)}"
            )

        for entry in entries:
            if category == "multi_turn":
                messages, criteria = entry

            else:
                prompt, criteria = entry

                messages = [
                    message(
                        "user",
                        prompt,
                    )
                ]

            add_qualitative(
                cases,
                language,
                category,
                messages,
                criteria,
            )

    qualitative_count = sum(
        1
        for case in cases
        if (
            case["language"]
            == language
            and case["kind"]
            == "qualitative"
        )
    )

    if qualitative_count != 40:
        raise RuntimeError(
            f"{language}: expected "
            f"40 qualitative cases, "
            f"got {qualitative_count}"
        )


# ============================================================
# LEAKAGE AUDIT
# ============================================================


def collect_corpus_prompts():
    prompts = set()

    for split in (
        "train",
        "validation",
        "test",
    ):
        rows = read_jsonl(
            DATA / f"{split}.jsonl"
        )

        for row in rows:
            for msg in row[
                "messages"
            ]:
                if msg["role"] != "user":
                    continue

                prompts.add(
                    normalize(
                        msg["content"]
                    )
                )

    return prompts


def benchmark_user_prompts(
    cases,
):
    prompts = []

    for case in cases:
        for msg in case[
            "messages"
        ]:
            if msg["role"] == "user":
                prompts.append(
                    normalize(
                        msg["content"]
                    )
                )

    return prompts


# ============================================================
# STRUCTURAL AUDIT
# ============================================================


def audit_cases(
    cases,
):
    if len(cases) != 300:
        raise RuntimeError(
            f"Expected 300 cases, "
            f"got {len(cases)}"
        )

    ids = [
        case["id"]
        for case in cases
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate benchmark IDs."
        )

    languages = Counter(
        case["language"]
        for case in cases
    )

    kinds = Counter(
        case["kind"]
        for case in cases
    )

    language_kind = Counter(
        (
            case["language"],
            case["kind"],
        )
        for case in cases
    )

    if languages != {
        "en": 100,
        "pt": 100,
        "es": 100,
    }:
        raise RuntimeError(
            f"Invalid language balance: "
            f"{dict(languages)}"
        )

    if kinds != {
        "objective": 180,
        "qualitative": 120,
    }:
        raise RuntimeError(
            f"Invalid benchmark kind balance: "
            f"{dict(kinds)}"
        )

    for language in (
        "en",
        "pt",
        "es",
    ):
        if (
            language_kind[
                (
                    language,
                    "objective",
                )
            ]
            != 60
        ):
            raise RuntimeError(
                f"{language}: objective "
                "count is not 60."
            )

        if (
            language_kind[
                (
                    language,
                    "qualitative",
                )
            ]
            != 40
        ):
            raise RuntimeError(
                f"{language}: qualitative "
                "count is not 40."
            )

    prompts = benchmark_user_prompts(
        cases
    )

    prompt_counts = Counter(
        prompts
    )

    duplicate_prompts = [
        prompt
        for prompt, count
        in prompt_counts.items()
        if count > 1
    ]

    if duplicate_prompts:
        raise RuntimeError(
            "Duplicate normalized benchmark "
            f"user prompts: "
            f"{len(duplicate_prompts)}"
        )

    corpus_prompts = (
        collect_corpus_prompts()
    )

    leakage = (
        set(prompts)
        & corpus_prompts
    )

    if leakage:
        print()
        print(
            "Benchmark/corpus prompt leakage:"
        )

        for prompt in sorted(
            leakage
        )[:20]:
            print(
                " ",
                prompt,
            )

        raise RuntimeError(
            f"Benchmark prompt leakage: "
            f"{len(leakage)}"
        )

    return {
        "languages":
            dict(languages),

        "kinds":
            dict(kinds),

        "language_kind": {
            f"{language}:{kind}":
                count
            for (
                language,
                kind,
            ), count
            in sorted(
                language_kind.items()
            )
        },

        "categories":
            dict(
                Counter(
                    case["category"]
                    for case in cases
                )
            ),

        "prompt_leakage":
            0,

        "duplicate_prompts":
            0,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print()
    print("=" * 80)
    print(
        "MAIA SFT VALIDATION"
    )
    print(
        "STAGE 4 V4 - "
        "DEFINITIVE ASSISTANT BENCHMARK"
    )
    print("=" * 80)

    cases = []

    # --------------------------------------------------------
    # Build deterministic benchmark
    # --------------------------------------------------------

    for language in (
        "en",
        "pt",
        "es",
    ):
        build_objective(
            cases,
            language,
        )

        build_qualitative(
            cases,
            language,
        )

    # --------------------------------------------------------
    # Deterministic canonical order
    # --------------------------------------------------------

    cases.sort(
        key=lambda case: (
            case["language"],
            case["kind"],
            case["id"],
        )
    )

    # --------------------------------------------------------
    # Full audit
    # --------------------------------------------------------

    print()
    print(
        "Auditing benchmark..."
    )

    audit = audit_cases(
        cases
    )

    # --------------------------------------------------------
    # Write benchmark
    # --------------------------------------------------------

    BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        BENCHMARK_FILE,
        cases,
    )

    digest = sha256_file(
        BENCHMARK_FILE
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "stage":
            "stage4-v4-definitive-assistant-benchmark",

        "seed":
            SEED_BENCHMARK,

        "benchmark_file":
            str(
                BENCHMARK_FILE.relative_to(
                    ROOT
                )
            ),

        "sha256":
            digest,

        "examples":
            len(cases),

        "languages": {
            "en": 100,
            "pt": 100,
            "es": 100,
        },

        "evaluation": {
            "objective":
                180,

            "qualitative":
                120,
        },

        "objective_per_language":
            60,

        "qualitative_per_language":
            40,

        "audit":
            audit,

        "corpus_independence": {
            "exact_normalized_user_prompt_overlap":
                0,
        },

        "status":
            "FROZEN_CANDIDATE",
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BENCHMARK COMPOSITION")
    print("=" * 80)

    print()
    print(
        "Examples:    ",
        len(cases),
    )

    print(
        "Objective:   ",
        sum(
            1
            for case in cases
            if case["kind"]
            == "objective"
        ),
    )

    print(
        "Qualitative: ",
        sum(
            1
            for case in cases
            if case["kind"]
            == "qualitative"
        ),
    )

    print()

    for language in (
        "en",
        "pt",
        "es",
    ):
        language_cases = [
            case
            for case in cases
            if (
                case["language"]
                == language
            )
        ]

        objective = sum(
            1
            for case
            in language_cases
            if (
                case["kind"]
                == "objective"
            )
        )

        qualitative = sum(
            1
            for case
            in language_cases
            if (
                case["kind"]
                == "qualitative"
            )
        )

        print(
            language.upper(),
            f"total={len(language_cases)}",
            f"objective={objective}",
            f"qualitative={qualitative}",
        )

    print()
    print(
        "Prompt leakage:    ",
        audit["prompt_leakage"],
    )

    print(
        "Duplicate prompts: ",
        audit["duplicate_prompts"],
    )

    print()
    print("=" * 80)
    print("BENCHMARK ARTIFACT")
    print("=" * 80)

    print()
    print(
        "File:"
    )

    print(
        " ",
        BENCHMARK_FILE,
    )

    print()
    print(
        "SHA-256:"
    )

    print(
        " ",
        digest,
    )

    print()
    print(
        "Manifest:"
    )

    print(
        " ",
        MANIFEST_FILE,
    )

    print()
    print("=" * 80)
    print(
        "STAGE 4 V4 BENCHMARK BUILD: PASS"
    )
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Do not start training yet."
    )

    print(
        "  Review this benchmark once."
    )

    print(
        "  After approval, freeze its SHA-256."
    )

    print(
        "  Then run the GPT-2 Medium baseline."
    )


if __name__ == "__main__":
    main()