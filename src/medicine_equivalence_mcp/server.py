import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from json import loads as json_loads
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from mcp.server.fastmcp import Context, FastMCP

from medicine_equivalence_mcp.dataset_loader import get_datasets_root

DATASETS_ROOT = get_datasets_root()
ITALY_DATA_PATH = DATASETS_ROOT / "Italy" / "confezioni_fornitura.csv"
FRANCE_DATA_DIR = DATASETS_ROOT / "France"
UK_DATA_ROOT = DATASETS_ROOT / "UK"
SPAIN_CIMA_API_BASE = "https://cima.aemps.es/cima/rest"

mcp = FastMCP("medicine-equivalence-server")
_REGISTRY_CACHE: dict[str, Any] | None = None
_UK_DF_CACHE: pd.DataFrame | None = None
PROMPTOPINION_FHIR_EXTENSION_NAME = "ai.promptopinion/fhir-context"


def _parse_scope_list(value: str) -> list[str]:
    """Parse a comma-separated scope list from environment configuration."""
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def get_promptopinion_requested_scopes() -> list[dict[str, Any]]:
    """Return PromptOpinion FHIR scopes requested by this MCP server.

    The default is an empty list because the medicine equivalence tools do not
    require direct patient-FHIR access to function.
    """
    requested = _parse_scope_list(os.getenv("PROMPTOPINION_FHIR_SCOPES", ""))
    required = set(_parse_scope_list(os.getenv("PROMPTOPINION_REQUIRED_FHIR_SCOPES", "")))
    return [
        {
            "name": scope_name,
            **({"required": True} if scope_name in required else {}),
        }
        for scope_name in requested
    ]


def get_promptopinion_fhir_context_from_ctx(ctx: Context) -> dict[str, Any]:
    """Read PromptOpinion FHIR context directly from request headers."""
    request = getattr(ctx.request_context, "request", None) if ctx.request_context is not None else None
    headers = getattr(request, "headers", {}) if request is not None else {}
    return {
        "fhir_server_url": headers.get("x-fhir-server-url"),
        "fhir_access_token": headers.get("x-fhir-access-token"),
        "patient_id": headers.get("x-patient-id"),
        "fhir_refresh_token": headers.get("x-fhir-refresh-token"),
        "fhir_refresh_url": headers.get("x-fhir-refresh-url"),
    }


def _install_promptopinion_fhir_extension(server: FastMCP[Any]) -> None:
    """Augment server capabilities with PromptOpinion's FHIR extension."""
    lowlevel_server = server._mcp_server
    if getattr(lowlevel_server, "_promptopinion_fhir_extension_installed", False):
        return

    original_get_capabilities = lowlevel_server.get_capabilities

    def get_capabilities_with_promptopinion_extension(notification_options: Any, experimental_capabilities: dict[str, dict[str, Any]]) -> Any:
        capabilities = original_get_capabilities(notification_options, experimental_capabilities)
        extensions = dict((capabilities.model_extra or {}).get("extensions", {}) or {})
        extensions[PROMPTOPINION_FHIR_EXTENSION_NAME] = {
            "scopes": get_promptopinion_requested_scopes(),
        }
        capabilities.model_extra["extensions"] = extensions
        return capabilities

    lowlevel_server.get_capabilities = get_capabilities_with_promptopinion_extension
    setattr(lowlevel_server, "_promptopinion_fhir_extension_installed", True)


_install_promptopinion_fhir_extension(mcp)


def _read_table_with_fallbacks(
    path: str | Path,
    *,
    sep: str,
    dtype: Any = str,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Read a delimited text file using a small set of practical encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, sep=sep, dtype=dtype, names=names, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to read file: {path}")


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """Perform a GET request and decode a UTF-8 JSON response."""
    query = urlencode({key: value for key, value in (params or {}).items() if value not in (None, "")})
    request_url = f"{url}?{query}" if query else url
    with urlopen(request_url) as response:
        return json_loads(response.read().decode("utf-8"))


def _extract_cima_items(payload: Any) -> list[dict[str, Any]]:
    """Extract a list of items from CIMA responses with tolerant shape handling."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("resultados", "medicamentos", "presentaciones", "lista", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _split_ingredient_names(value: Any) -> list[str]:
    """Split ingredient labels into normalized parts usable for fallback searches."""
    text = str(value or "")
    parts = re.split(r"\s*(?:/|\+|,|;|\by\b|\bet\b|\band\b)\s*", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if normalize_text(part.strip())]


def build_direct_ingredient_atc_map(*dfs: pd.DataFrame | None) -> dict[str, str]:
    """Build an exact ingredient -> ATC map from direct country datasets only.

    Only ingredient keys with a single unambiguous ATC across all provided
    direct-source datasets are kept.
    """
    ingredient_to_atcs: dict[str, set[str]] = {}

    for df in dfs:
        if df is None or df.empty:
            continue
        direct_rows = df[
            df["atc"].astype(str).str.strip().ne("")
            & df["ingredients_norm"].astype(str).str.strip().ne("")
        ][["ingredients_norm", "atc"]].drop_duplicates()
        for _, row in direct_rows.iterrows():
            ingredient = str(row["ingredients_norm"]).strip()
            atc = str(row["atc"]).strip()
            if ingredient and atc:
                ingredient_to_atcs.setdefault(ingredient, set()).add(atc)

    return {
        ingredient: next(iter(atcs))
        for ingredient, atcs in ingredient_to_atcs.items()
        if len(atcs) == 1
    }


def apply_ingredient_atc_backfill(df: pd.DataFrame, ingredient_atc_map: dict[str, str]) -> pd.DataFrame:
    """Backfill ATC using an exact normalized-ingredient map."""
    if df.empty or not ingredient_atc_map:
        return df

    filled = df.copy()
    missing_mask = filled["atc"].eq("") & filled["ingredients_norm"].ne("")
    inferred = filled.loc[missing_mask, "ingredients_norm"].map(ingredient_atc_map).fillna("")
    filled.loc[missing_mask, "atc"] = inferred
    filled.loc[
        missing_mask & filled["atc"].ne("") & filled["atc_source"].eq("missing"),
        "atc_source",
    ] = "ingredient_backfill"
    return filled


def apply_same_country_atc_consensus_backfill(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Backfill missing ATC from strong same-country consensus keys.

    This is intentionally conservative. It only fills ATC when a key maps to a
    single unambiguous ATC inside the same country dataset.

    Priority:
    1. exact ingredient + dosage + normalized form
    2. exact ingredient + dosage
    3. exact ingredient only
    """
    if df.empty:
        return df

    filled = df.copy()
    direct_rows = filled[
        filled["atc"].astype(str).str.strip().ne("")
        & filled["ingredients_norm"].astype(str).str.strip().ne("")
    ].copy()
    if direct_rows.empty:
        return filled

    def _single_atc_map(group_cols: list[str]) -> pd.Series:
        grouped = (
            direct_rows.groupby(group_cols)["atc"]
            .agg(lambda values: sorted({str(value).strip() for value in values if str(value).strip()}))
        )
        return grouped[grouped.apply(len) == 1].apply(lambda values: values[0])

    exact_key_map = _single_atc_map(["ingredients_norm", "dosage_mg", "form_norm"])
    dosage_key_map = _single_atc_map(["ingredients_norm", "dosage_mg"])
    ingredient_key_map = _single_atc_map(["ingredients_norm"])

    missing_mask = filled["atc"].eq("") & filled["ingredients_norm"].ne("")
    if not missing_mask.any():
        return filled

    country_label = normalize_text(country) or "country"

    # Use the strongest exact-match key first.
    for index in filled.index[missing_mask]:
        row = filled.loc[index]

        exact_key = (row["ingredients_norm"], row["dosage_mg"], row["form_norm"])
        dosage_key = (row["ingredients_norm"], row["dosage_mg"])
        ingredient_key = row["ingredients_norm"]

        inferred_atc = ""
        inferred_source = ""

        if exact_key in exact_key_map.index:
            inferred_atc = exact_key_map.loc[exact_key]
            inferred_source = f"{country_label}_same_country_exact"
        elif dosage_key in dosage_key_map.index:
            inferred_atc = dosage_key_map.loc[dosage_key]
            inferred_source = f"{country_label}_same_country_dosage"
        elif ingredient_key in ingredient_key_map.index:
            inferred_atc = ingredient_key_map.loc[ingredient_key]
            inferred_source = f"{country_label}_same_country_ingredient"

        if inferred_atc:
            filled.at[index, "atc"] = inferred_atc
            if filled.at[index, "atc_source"] == "missing":
                filled.at[index, "atc_source"] = inferred_source

    return filled


def _get_uk_extract_dir(root: str | Path = UK_DATA_ROOT) -> Path:
    """Return the extracted NHS dm+d directory from the UK dataset root."""
    root_path = Path(root)
    candidates = [path for path in root_path.iterdir() if path.is_dir() and path.name.startswith("nhsbsa_dmd_")]
    if not candidates:
        raise FileNotFoundError(f"No extracted UK dm+d directory found under {root_path}")
    return sorted(candidates)[-1]


def _get_uk_bonus_dir(root: str | Path = UK_DATA_ROOT) -> Path:
    """Return the extracted NHS dm+d bonus directory from the UK dataset root."""
    root_path = Path(root)
    candidates = [path for path in root_path.iterdir() if path.is_dir() and path.name.startswith("nhsbsa_dmdbonus_")]
    if not candidates:
        raise FileNotFoundError(f"No extracted UK dm+d bonus directory found under {root_path}")
    return sorted(candidates)[-1]


def _xml_text(element: ET.Element, child_name: str) -> str:
    """Return stripped child text from an XML element."""
    child = element.find(child_name)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _strip_accents(text: str) -> str:
    """Return an ASCII-like representation suitable for loose matching."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: Any) -> str:
    """Normalize free text for case-insensitive matching across datasets."""
    if pd.isna(value):
        return ""
    text = _strip_accents(str(value)).lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    ingredient_synonyms = {
        "paracetamolo": "paracetamol",
        "acetaminophen": "paracetamol",
    }
    for source, target in ingredient_synonyms.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)

    return text


def extract_dosage_mg(value: Any) -> float | None:
    """Extract the first dosage from text and convert it to milligrams.

    Examples:
    - "500 mg" -> 500.0
    - "1 g" -> 1000.0
    - "800 microgrammes" -> 0.8
    """
    if pd.isna(value):
        return None

    text = _strip_accents(str(value)).lower().replace(",", ".")
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(mg|g|grammes?|grammi?|mcg|ug|µg|microgrammes?|microgrammi?)\b"
    )
    match = pattern.search(text)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2)

    if unit in {"g", "gramme", "grammes", "grammo", "grammi"}:
        return amount * 1000.0
    if unit in {"mcg", "ug", "µg", "microgramme", "microgrammes", "microgrammo", "microgrammi"}:
        return amount / 1000.0
    return amount


def _coerce_float(value: Any) -> float | None:
    """Safely coerce a sparse dosage-like value to float."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _normalize_form(value: Any) -> str:
    """Normalize pharmaceutical form labels to improve cross-country matching."""
    text = normalize_text(value)
    replacements = [
        ("compresse rivestite con film", "tablet"),
        ("compressa rivestita con film", "tablet"),
        ("compressa rivestita", "tablet"),
        ("compresse rivestite", "tablet"),
        ("comprime pellicule secable", "tablet"),
        ("comprime pellicule", "tablet"),
        ("comprime a liberation prolongee", "tablet"),
        ("comprime secable", "tablet"),
        ("comprime effervescent", "tablet"),
        ("comprime orodispersible", "tablet"),
        ("comprime", "tablet"),
        ("compresse", "tablet"),
        ("compressa", "tablet"),
        ("tablets", "tablet"),
        ("tablet", "tablet"),
        ("poudre pour solution buvable", "poudre"),
        ("capsule molle", "capsule"),
        ("gelule", "capsule"),
    ]
    for source, target in replacements:
        text = text.replace(source, target)
    return text.strip()


def _extract_query_form_hint(value: Any) -> str:
    """Extract a normalized dosage-form hint from a free-text user query."""
    text = normalize_text(value)
    form_aliases = {
        "tablet": "tablet",
        "tablets": "tablet",
        "tab": "tablet",
        "tabs": "tablet",
        "pill": "tablet",
        "pills": "tablet",
        "compressa": "tablet",
        "compresse": "tablet",
        "comprime": "tablet",
        "capsule": "capsule",
        "capsules": "capsule",
        "capsula": "capsule",
        "capsule molle": "capsule",
        "syrup": "sciroppo",
        "sciroppo": "sciroppo",
        "sirup": "sciroppo",
        "suspension": "sospensione",
        "sospensione": "sospensione",
        "solution": "soluzione",
        "soluzione": "soluzione",
        "drops": "gocce",
        "gocce": "gocce",
        "suppository": "supposta",
        "suppositories": "supposta",
        "supposta": "supposta",
        "bustina": "bustina",
        "bustine": "bustina",
        "sachet": "bustina",
        "sachets": "bustina",
    }

    for alias, normalized_form in sorted(form_aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text.split() or alias in text:
            return normalized_form
    return ""


def _build_search_text(*values: Any) -> str:
    """Join several fields into a normalized search string."""
    return " ".join(part for part in (normalize_text(value) for value in values) if part).strip()


def describe_atc_source(atc_source: Any) -> str:
    """Return a user-facing explanation of how the ATC value was obtained."""
    source = str(atc_source or "missing").strip()
    descriptions = {
        "direct_dataset": "ATC was read directly from the source dataset record.",
        "bonus_bnf": "ATC was filled from the UK bonus BNF mapping dataset.",
        "ingredient_backfill": "ATC was inferred from an exact ingredient-level mapping built from direct dataset records.",
        "france_same_country_exact": "ATC was inferred from another France record with the same ingredient, dosage, and form.",
        "france_same_country_dosage": "ATC was inferred from another France record with the same ingredient and dosage.",
        "france_same_country_ingredient": "ATC was inferred from another France record with the same ingredient.",
        "missing": "No ATC could be inferred from the available data.",
    }
    return descriptions.get(source, f"ATC source: {source}.")


def summarize_access_info(access_info: Any) -> dict[str, Any]:
    """Return a simplified user-facing summary of access restrictions."""
    raw_text = str(access_info or "").strip()
    norm = normalize_text(raw_text)

    if not norm:
        return {
            "restriction_status": "unknown",
            "simple_message": "Restriction information is not available.",
            "raw_access_info": None,
        }

    prescription_markers = [
        "prescription",
        "ricetta",
        "requiere receta",
        "receta",
        "liste i",
        "liste ii",
        "subject to medical prescription",
        "soggetti a prescrizione medica",
    ]
    otc_markers = [
        "no requiere receta",
        "sans ordonnance",
        "without prescription",
        "otc",
        "over the counter",
        "senza prescrizione",
        "prescription not required",
    ]

    if any(marker in norm for marker in prescription_markers):
        return {
            "restriction_status": "prescription_required",
            "simple_message": "This medicine appears to require a prescription.",
            "raw_access_info": raw_text,
        }

    if any(marker in norm for marker in otc_markers):
        return {
            "restriction_status": "otc_or_no_prescription",
            "simple_message": "This medicine appears to be available without a prescription.",
            "raw_access_info": raw_text,
        }

    return {
        "restriction_status": "restriction_unclear",
        "simple_message": "There may be dispensing restrictions; check the detailed access information.",
        "raw_access_info": raw_text,
    }


def build_user_summary(
    resolved_source: dict[str, Any] | None,
    matches: list[dict[str, Any]],
    total_equivalents: int,
) -> dict[str, Any]:
    """Build a simple top-level summary for non-specialist users."""
    if not resolved_source:
        return {
            "can_consider_equivalent": False,
            "summary": "The source medicine could not be resolved.",
            "match_level": "unresolved",
            "restriction_note": "Restriction information is not available.",
            "safety_note": "No equivalent suggestion can be made until the source medicine is identified.",
        }

    if total_equivalents == 0 or not matches:
        return {
            "can_consider_equivalent": False,
            "summary": "No likely equivalent was found in the destination country.",
            "match_level": "none",
            "restriction_note": resolved_source.get("access_summary", {}).get(
                "simple_message",
                "Restriction information is not available.",
            ),
            "safety_note": "Do not substitute this medicine without pharmacist or prescriber review.",
        }

    best_match = matches[0]
    match_details = best_match.get("match_details", {})
    exact_equivalent = bool(
        match_details.get("atc_match")
        and match_details.get("dosage_match")
        and match_details.get("form_match")
    )
    close_equivalent = bool(
        match_details.get("atc_match") or match_details.get("ingredient_match")
    )

    if exact_equivalent:
        match_level = "exact"
        summary = "A close equivalent was found with the same drug class, dosage, and form."
        safety_note = "Even with a close match, confirm local substitution rules before switching."
    elif close_equivalent:
        match_level = "close"
        summary = "A similar medicine was found, but there are differences to review."
        safety_note = "Check dosage, form, and access conditions before substituting."
    else:
        match_level = "broad"
        summary = "Only broader ingredient-based alternatives were found."
        safety_note = "A pharmacist or prescriber should confirm whether substitution is appropriate."

    restriction_messages = []
    source_restriction = resolved_source.get("access_summary", {}).get("simple_message")
    if source_restriction:
        restriction_messages.append(f"Source: {source_restriction}")
    destination_restriction = best_match.get("access_summary", {}).get("simple_message")
    if destination_restriction:
        restriction_messages.append(f"Best match: {destination_restriction}")

    return {
        "can_consider_equivalent": close_equivalent,
        "summary": summary,
        "match_level": match_level,
        "restriction_note": " ".join(restriction_messages) if restriction_messages else "Restriction information is not available.",
        "safety_note": safety_note,
    }


def _extract_brand_name(value: Any) -> str:
    """Extract a practical brand label from a marketed drug denomination."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    head = text.split(",")[0].strip()
    head = re.sub(
        r"\b\d+(?:[.,]\d+)?\s*(mg|g|mcg|ug|µg|ui|ml|mg\/ml|g\/ml|%)\b.*$",
        "",
        head,
        flags=re.IGNORECASE,
    ).strip()

    tokens = head.split()
    brand_tokens: list[str] = []
    for token in tokens:
        token_norm = normalize_text(token)
        if not token_norm:
            continue
        if re.search(r"\d", token):
            break
        if token_norm in {
            "mg",
            "g",
            "ml",
            "ui",
            "pour",
            "cent",
            "comprime",
            "capsule",
            "gelule",
            "solution",
            "suspension",
            "sciroppo",
            "tablet",
        }:
            break
        brand_tokens.append(token)

    return " ".join(brand_tokens).strip() or head


def _select_best_row(matches: pd.DataFrame, query_norm: str) -> pd.Series:
    """Pick the best matching row from a candidate set."""
    query_tokens = set(query_norm.split())
    query_dosage = extract_dosage_mg(query_norm)
    query_form_hint = _extract_query_form_hint(query_norm)

    scored = matches.copy()
    scored["exact_name_match"] = scored["search_text"].eq(query_norm).astype(int)
    scored["starts_with_match"] = scored["search_text"].str.startswith(query_norm).fillna(False).astype(int)
    scored["token_overlap"] = scored["search_text"].apply(
        lambda text: len(query_tokens.intersection(set(str(text).split())))
    )
    scored["query_dosage_match"] = scored["dosage_mg"].apply(
        lambda value: int(
            query_dosage is not None
            and _coerce_float(value) is not None
            and _coerce_float(value) == float(query_dosage)
        )
    )
    scored["query_form_match"] = scored["form_norm"].apply(
        lambda value: int(bool(query_form_hint) and query_form_hint in str(value))
    )
    scored["query_dosage_delta"] = scored["dosage_mg"].apply(
        lambda value: abs(_coerce_float(value) - float(query_dosage))
        if query_dosage is not None and _coerce_float(value) is not None
        else float("inf")
    )
    scored["has_atc"] = scored["atc"].ne("").astype(int)
    scored["has_dosage"] = scored["dosage_mg"].apply(lambda value: int(_coerce_float(value) is not None))

    scored = scored.sort_values(
        by=[
            "exact_name_match",
            "query_dosage_match",
            "query_form_match",
            "starts_with_match",
            "token_overlap",
            "has_atc",
            "has_dosage",
            "query_dosage_delta",
        ],
        ascending=[False, False, False, False, False, False, False, True],
    )
    return scored.iloc[0]


def _record_to_resolved_dict(record: pd.Series, drug_query: str, country: str) -> dict[str, Any]:
    """Convert a normalized dataframe record into the response payload."""
    return {
        "query": drug_query,
        "country": country,
        "drug_name": record["drug_name"],
        "brand_name": record["brand_name"],
        "ingredients": record["ingredients"],
        "atc": record["atc"] or None,
        "atc_source": record.get("atc_source", "missing"),
        "atc_inference": describe_atc_source(record.get("atc_source", "missing")),
        "dosage_mg": _coerce_float(record["dosage_mg"]),
        "form": record["form"] or None,
        "access_info": record["access_info"] or None,
        "access_summary": summarize_access_info(record["access_info"]),
        "matched_on": record.get("matched_on"),
        "source_record": {
            "record_id": record["record_id"],
            "source_country": record["source_country"],
        },
    }


def _candidate_variant_summary(matches: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    """Summarize distinct source variants when a query maps to multiple strengths or forms."""
    summary = (
        matches[["drug_name", "ingredients", "atc", "dosage_mg", "form", "access_info"]]
        .drop_duplicates()
        .head(limit)
    )

    results: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        results.append(
            {
                "drug_name": row["drug_name"],
                "ingredients": row["ingredients"] or None,
                "atc": row["atc"] or None,
                "dosage_mg": _coerce_float(row["dosage_mg"]),
                "form": row["form"] or None,
                "access_info": row["access_info"] or None,
            }
        )
    return results


def _get_brand_family_candidates(df: pd.DataFrame, drug_query: str) -> pd.DataFrame:
    """Return rows that appear to belong to the queried brand family."""
    brand_query = normalize_text(_extract_brand_name(drug_query))
    if not brand_query:
        return df.iloc[0:0].copy()

    brand_mask = (
        df["brand_name_norm"].eq(brand_query)
        | df["brand_name_norm"].str.startswith(f"{brand_query} ").fillna(False)
        | df["drug_name_norm"].str.startswith(f"{brand_query} ").fillna(False)
    )
    return df[brand_mask].copy()


def load_italy_aifa(path: str | Path) -> pd.DataFrame:
    """Load the Italy AIFA file and standardize it for cross-country matching."""
    df = _read_table_with_fallbacks(path, sep=";", dtype=str)
    df = df.fillna("")

    normalized = pd.DataFrame(
        {
            "record_id": df["CODICE_AIC"].astype(str),
            "source_country": "Italy",
            "drug_name": df["DENOMINAZIONE"].astype(str).str.strip(),
            "brand_name": df["DENOMINAZIONE"].astype(str).apply(_extract_brand_name),
            "ingredients": df["PA_ASSOCIATI"].astype(str).str.strip(),
            "atc": df["CODICE_ATC"].astype(str).str.strip(),
            "form": df["FORMA"].astype(str).str.strip(),
            "access_info": df["FORNITURA"].astype(str).str.strip(),
        }
    )

    dosage_source = (
        df["DENOMINAZIONE"].astype(str)
        + " "
        + df.get("DESCRIZIONE", pd.Series([""] * len(df))).astype(str)
    )
    normalized["dosage_mg"] = dosage_source.apply(extract_dosage_mg)
    normalized["form_norm"] = normalized["form"].apply(_normalize_form)
    normalized["drug_name_norm"] = normalized["drug_name"].apply(normalize_text)
    normalized["brand_name_norm"] = normalized["brand_name"].apply(normalize_text)
    normalized["ingredients_norm"] = normalized["ingredients"].apply(normalize_text)
    normalized["atc_source"] = normalized["atc"].apply(lambda value: "direct_dataset" if str(value).strip() else "missing")
    normalized["search_text"] = normalized.apply(
        lambda row: _build_search_text(row["drug_name"], row["ingredients"]),
        axis=1,
    )

    return normalized


def load_france_bdpm(
    cis_path: str | Path,
    compo_path: str | Path,
    cpd_path: str | Path,
    mitm_path: str | Path,
) -> pd.DataFrame:
    """Load the France BDPM files and merge them in memory.

    The function keeps the source files separate on disk and only joins them into
    a normalized dataframe for runtime use.
    """
    cis_columns = [
        "CIS",
        "DENOMINATION",
        "FORME",
        "VOIES",
        "STATUT_AMM",
        "TYPE_PROCEDURE",
        "ETAT_COMMERCIALISATION",
        "DATE_AMM",
        "STATUT_BDM",
        "NUMERO_EU",
        "TITULAIRE",
        "SURVEILLANCE_RENFORCEE",
    ]
    compo_columns = [
        "CIS",
        "FORME_COMPOSANT",
        "CODE_SUBSTANCE",
        "SUBSTANCE",
        "DOSAGE",
        "REFERENCE_DOSAGE",
        "NATURE_COMPOSANT",
        "NUMERO_LIAISON",
    ]
    cpd_columns = ["CIS", "ACCESS_INFO"]
    mitm_columns = ["CIS", "ATC", "DENOMINATION_COURTE", "URL"]

    cis = _read_table_with_fallbacks(cis_path, sep="\t", names=cis_columns, dtype=str).fillna("")
    compo = _read_table_with_fallbacks(compo_path, sep="\t", names=compo_columns, dtype=str).fillna("")
    cpd = _read_table_with_fallbacks(cpd_path, sep="\t", names=cpd_columns, dtype=str).fillna("")
    mitm = _read_table_with_fallbacks(mitm_path, sep="\t", names=mitm_columns, dtype=str).fillna("")

    compo_agg = (
        compo.groupby("CIS", as_index=False)
        .agg(
            ingredients=("SUBSTANCE", lambda values: " + ".join(sorted({v.strip() for v in values if v.strip()}))),
            dosage_text=("DOSAGE", lambda values: " | ".join(sorted({v.strip() for v in values if v.strip()}))),
        )
    )
    cpd_agg = (
        cpd.groupby("CIS", as_index=False)
        .agg(access_info=("ACCESS_INFO", lambda values: " | ".join(sorted({v.strip() for v in values if v.strip()}))))
    )

    merged = cis.merge(compo_agg, on="CIS", how="left")
    merged = merged.merge(cpd_agg, on="CIS", how="left")
    merged = merged.merge(mitm[["CIS", "ATC", "URL"]], on="CIS", how="left")
    merged = merged.fillna("")

    normalized = pd.DataFrame(
        {
            "record_id": merged["CIS"].astype(str),
            "source_country": "France",
            "drug_name": merged["DENOMINATION"].astype(str).str.strip(),
            "brand_name": merged["DENOMINATION"].astype(str).apply(_extract_brand_name),
            "ingredients": merged["ingredients"].astype(str).str.strip(),
            "atc": merged["ATC"].astype(str).str.strip(),
            "form": merged["FORME"].astype(str).str.strip(),
            "access_info": merged["access_info"].astype(str).str.strip(),
            "url": merged["URL"].astype(str).str.strip(),
        }
    )

    dosage_source = merged["DENOMINATION"].astype(str) + " " + merged["dosage_text"].astype(str)
    normalized["dosage_mg"] = dosage_source.apply(extract_dosage_mg)
    normalized["form_norm"] = normalized["form"].apply(_normalize_form)
    normalized["drug_name_norm"] = normalized["drug_name"].apply(normalize_text)
    normalized["brand_name_norm"] = normalized["brand_name"].apply(normalize_text)
    normalized["ingredients_norm"] = normalized["ingredients"].apply(normalize_text)
    normalized["atc_source"] = normalized["atc"].apply(lambda value: "direct_dataset" if str(value).strip() else "missing")
    normalized["search_text"] = normalized.apply(
        lambda row: _build_search_text(row["drug_name"], row["ingredients"]),
        axis=1,
    )

    return normalized


def _normalize_spain_medication_detail(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a single CIMA medicamento detail response into normalized rows."""
    nregistro = str(detail.get("nregistro", "")).strip()
    drug_name = str(detail.get("nombre", "")).strip()
    brand_name = _extract_brand_name(drug_name)
    active_ingredients = detail.get("principiosActivos") or []
    ingredient_names = [str(item.get("nombre", "")).strip() for item in active_ingredients if item.get("nombre")]
    ingredients = " + ".join(dict.fromkeys(ingredient_names))

    atc_items = detail.get("atcs") or []
    atc_code = ""
    if atc_items:
        atc_code = str(atc_items[0].get("codigo", "")).strip()

    form_info = detail.get("formaFarmaceuticaSimplificada") or detail.get("formaFarmaceutica") or {}
    form = str(form_info.get("nombre", "")).strip()
    dosage_text = str(detail.get("dosis", "")).strip()
    access_parts: list[str] = []
    if detail.get("cpresc"):
        access_parts.append(str(detail.get("cpresc")).strip())
    if detail.get("receta") is True:
        access_parts.append("Requiere receta")
    elif detail.get("receta") is False:
        access_parts.append("No requiere receta")
    access_info = " | ".join(dict.fromkeys(part for part in access_parts if part))

    presentation_rows: list[dict[str, Any]] = []
    for presentation in detail.get("presentaciones") or []:
        presentation_name = str(presentation.get("nombre", "")).strip()
        presentation_cn = str(presentation.get("cn", "")).strip()
        row_name = presentation_name or drug_name
        row_access = access_info
        if presentation.get("cpresc"):
            row_access = " | ".join(
                dict.fromkeys([part for part in [str(presentation.get("cpresc")).strip(), access_info] if part])
            )
        presentation_rows.append(
            {
                "record_id": presentation_cn or nregistro,
                "source_country": "Spain",
                "drug_name": row_name,
                "brand_name": brand_name or _extract_brand_name(row_name),
                "ingredients": ingredients,
                "atc": atc_code,
                "form": form or row_name,
                "access_info": row_access,
            }
        )

    if not presentation_rows:
        presentation_rows.append(
            {
                "record_id": nregistro,
                "source_country": "Spain",
                "drug_name": drug_name,
                "brand_name": brand_name,
                "ingredients": ingredients,
                "atc": atc_code,
                "form": form,
                "access_info": access_info,
            }
        )

    for row in presentation_rows:
        dosage_source = " ".join(part for part in [row["drug_name"], dosage_text] if part)
        row["dosage_mg"] = extract_dosage_mg(dosage_source)
        row["form_norm"] = _normalize_form(row["form"])
        row["drug_name_norm"] = normalize_text(row["drug_name"])
        row["brand_name_norm"] = normalize_text(row["brand_name"])
        row["ingredients_norm"] = normalize_text(row["ingredients"])
        row["atc_source"] = "direct_dataset" if row["atc"] else "missing"
        row["search_text"] = _build_search_text(row["drug_name"], row["brand_name"], row["ingredients"])

    return presentation_rows


def _cima_fetch_medicamento_detail(nregistro: str, api_base: str = SPAIN_CIMA_API_BASE) -> dict[str, Any]:
    """Fetch the full CIMA medicamento detail by registration number."""
    return _http_get_json(f"{api_base}/medicamento", {"nregistro": nregistro})


def load_spain_cima(
    *,
    api_base: str = SPAIN_CIMA_API_BASE,
    nombre: str | None = None,
    atc: str | None = None,
    practiv1: str | None = None,
    practiv2: str | None = None,
    receta: int | None = None,
    comerc: int = 1,
    autorizados: int = 1,
    max_results: int = 100,
) -> pd.DataFrame:
    """Load Spain medicines from the CIMA REST API and normalize them for matching.

    The function searches CIMA's `medicamentos` endpoint, then fetches the full
    `medicamento` detail for each returned registration number in order to obtain
    ATC, active ingredients, form, dosage and prescription metadata.
    """
    params: dict[str, Any] = {
        "nombre": nombre,
        "atc": atc,
        "practiv1": practiv1,
        "practiv2": practiv2,
        "receta": receta,
        "comerc": comerc,
        "autorizados": autorizados,
    }
    payload = _http_get_json(f"{api_base}/medicamentos", params)
    items = _extract_cima_items(payload)
    if not items and isinstance(payload, dict) and payload.get("nregistro"):
        items = [payload]

    rows: list[dict[str, Any]] = []
    for item in items[:max_results]:
        nregistro = str(item.get("nregistro", "")).strip()
        if not nregistro:
            continue
        detail = _cima_fetch_medicamento_detail(nregistro, api_base=api_base)
        rows.extend(_normalize_spain_medication_detail(detail))

    if not rows:
        return pd.DataFrame(
            columns=[
                "record_id",
                "source_country",
                "drug_name",
                "brand_name",
                "ingredients",
                "atc",
                "form",
                "access_info",
                "dosage_mg",
                "form_norm",
                "drug_name_norm",
                "brand_name_norm",
                "ingredients_norm",
                "search_text",
            ]
        )

    normalized = pd.DataFrame(rows).fillna("")
    return normalized.drop_duplicates(subset=["record_id", "drug_name", "atc", "form", "dosage_mg"]).reset_index(
        drop=True
    )


def _load_uk_lookup_maps(extract_dir: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load UK dm+d form and legal-category lookup dictionaries."""
    lookup_path = Path(extract_dir) / next(
        name for name in Path(extract_dir).iterdir() if name.name.startswith("f_lookup2_") and name.suffix == ".xml"
    ).name
    root = ET.parse(lookup_path).getroot()

    form_map: dict[str, str] = {}
    legal_map: dict[str, str] = {}

    form_section = root.find("FORM")
    if form_section is not None:
        for info in form_section.findall("INFO"):
            code = _xml_text(info, "CD")
            desc = _xml_text(info, "DESC")
            if code:
                form_map[code] = desc

    legal_section = root.find("LEGAL_CATEGORY")
    if legal_section is not None:
        for info in legal_section.findall("INFO"):
            code = _xml_text(info, "CD")
            desc = _xml_text(info, "DESC")
            if code:
                legal_map[code] = desc

    return form_map, legal_map


def _load_uk_bonus_atc_map(bonus_dir: str | Path | None = None) -> dict[str, str]:
    """Load direct UK VPID -> ATC mappings from the dm+d bonus BNF extract."""
    bonus_root = Path(bonus_dir) if bonus_dir is not None else _get_uk_bonus_dir()
    bnf_zip = next(path for path in bonus_root.iterdir() if path.name.endswith("-BNF.zip"))

    atc_map: dict[str, str] = {}
    with zipfile.ZipFile(bnf_zip) as archive:
        xml_name = next(name for name in archive.namelist() if name.startswith("f_bnf") and name.endswith(".xml"))
        with archive.open(xml_name) as xml_file:
            for _, elem in ET.iterparse(xml_file, events=("end",)):
                if elem.tag != "VMP":
                    continue
                vpid = _xml_text(elem, "VPID")
                atc = _xml_text(elem, "ATC")
                if vpid and atc and atc.lower() != "n/a":
                    atc_map[vpid] = atc
                elem.clear()

    return atc_map


def load_uk_dmd(
    extract_dir: str | Path | None = None,
    bonus_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load UK NHS dm+d XML extracts and normalize them for matching.

    The loader uses packaged products (AMPP) for brand-facing UK rows, joins them
    back to AMP and VMP records for ingredients and form, then backfills ATC later
    from the other country datasets because dm+d does not expose ATC in these files.
    """
    uk_dir = Path(extract_dir) if extract_dir is not None else _get_uk_extract_dir()
    form_lookup, legal_lookup = _load_uk_lookup_maps(uk_dir)
    atc_by_vpid = _load_uk_bonus_atc_map(bonus_dir)

    ingredient_path = next(path for path in uk_dir.iterdir() if path.name.startswith("f_ingredient2_") and path.suffix == ".xml")
    vmp_path = next(path for path in uk_dir.iterdir() if path.name.startswith("f_vmp2_") and path.suffix == ".xml")
    amp_path = next(path for path in uk_dir.iterdir() if path.name.startswith("f_amp2_") and path.suffix == ".xml")
    ampp_path = next(path for path in uk_dir.iterdir() if path.name.startswith("f_ampp2_") and path.suffix == ".xml")

    ingredient_map: dict[str, str] = {}
    for _, elem in ET.iterparse(ingredient_path, events=("end",)):
        if elem.tag == "ING":
            isid = _xml_text(elem, "ISID")
            name = _xml_text(elem, "NM")
            if isid:
                ingredient_map[isid] = name
            elem.clear()

    vpid_to_name: dict[str, str] = {}
    vpid_to_form: dict[str, str] = {}
    vpid_to_ingredients: dict[str, list[str]] = {}
    for _, elem in ET.iterparse(vmp_path, events=("end",)):
        if elem.tag == "VMP":
            if _xml_text(elem, "INVALID") == "1":
                elem.clear()
                continue
            vpid = _xml_text(elem, "VPID")
            if vpid:
                vpid_to_name[vpid] = _xml_text(elem, "NM")
            elem.clear()
        elif elem.tag == "DFORM":
            vpid = _xml_text(elem, "VPID")
            form_code = _xml_text(elem, "FORMCD")
            if vpid and form_code:
                vpid_to_form[vpid] = form_lookup.get(form_code, form_code)
            elem.clear()
        elif elem.tag == "VPI":
            vpid = _xml_text(elem, "VPID")
            ingredient_id = _xml_text(elem, "ISID")
            ingredient_name = ingredient_map.get(ingredient_id, "")
            if vpid and ingredient_name:
                vpid_to_ingredients.setdefault(vpid, [])
                if ingredient_name not in vpid_to_ingredients[vpid]:
                    vpid_to_ingredients[vpid].append(ingredient_name)
            elem.clear()

    apid_to_vpid: dict[str, str] = {}
    for _, elem in ET.iterparse(amp_path, events=("end",)):
        if elem.tag == "AMP":
            if _xml_text(elem, "INVALID") == "1":
                elem.clear()
                continue
            apid = _xml_text(elem, "APID")
            vpid = _xml_text(elem, "VPID")
            if apid and vpid:
                apid_to_vpid[apid] = vpid
            elem.clear()

    rows: list[dict[str, Any]] = []
    for _, elem in ET.iterparse(ampp_path, events=("end",)):
        if elem.tag != "AMPP":
            continue

        if _xml_text(elem, "INVALID") == "1":
            elem.clear()
            continue

        appid = _xml_text(elem, "APPID")
        apid = _xml_text(elem, "APID")
        vpid = apid_to_vpid.get(apid, "")
        drug_name = _xml_text(elem, "NM")
        ingredients = " + ".join(vpid_to_ingredients.get(vpid, []))
        form = vpid_to_form.get(vpid) or vpid_to_name.get(vpid, "")
        legal_cat = legal_lookup.get(_xml_text(elem, "LEGAL_CATCD"), _xml_text(elem, "LEGAL_CATCD"))

        if not appid or not drug_name:
            elem.clear()
            continue

        row = {
            "record_id": appid,
            "source_country": "UK",
            "drug_name": drug_name,
            "brand_name": _extract_brand_name(drug_name),
            "ingredients": ingredients,
            "atc": atc_by_vpid.get(vpid, ""),
            "form": form,
            "access_info": legal_cat,
        }
        row["dosage_mg"] = extract_dosage_mg(drug_name)
        row["form_norm"] = _normalize_form(row["form"])
        row["drug_name_norm"] = normalize_text(row["drug_name"])
        row["brand_name_norm"] = normalize_text(row["brand_name"])
        row["ingredients_norm"] = normalize_text(row["ingredients"])
        row["atc_source"] = "bonus_bnf" if row["atc"] else "missing"
        row["search_text"] = _build_search_text(row["drug_name"], row["brand_name"], row["ingredients"])
        rows.append(row)
        elem.clear()

    normalized = pd.DataFrame(rows).fillna("")
    return normalized.drop_duplicates(subset=["record_id", "drug_name"]).reset_index(drop=True)


def build_country_registry(
    df_it: pd.DataFrame,
    df_fr: pd.DataFrame,
    df_uk: pd.DataFrame | None = None,
    df_who: pd.DataFrame | None = None,
    spain_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lookup registry by normalized country name."""
    _ = df_who
    ingredient_atc_map = build_direct_ingredient_atc_map(df_it, df_fr, df_uk)
    france_df = apply_same_country_atc_consensus_backfill(df_fr.copy(), "France")
    france_df = apply_ingredient_atc_backfill(france_df, ingredient_atc_map)

    registry: dict[str, Any] = {
        "italy": df_it.copy(),
        "france": france_df,
        "spain": {
            "type": "cima_api",
            "api_base": SPAIN_CIMA_API_BASE,
            "ingredient_atc_map": ingredient_atc_map,
            **(spain_config or {}),
        },
    }

    if df_uk is not None and not df_uk.empty:
        registry["uk"] = df_uk.copy()

    return registry


def resolve_in_country(df: pd.DataFrame, drug_query: str) -> dict[str, Any] | None:
    """Resolve a drug query inside a country dataframe.

    The resolver supports both brand-name and ingredient/generic queries and uses
    loose normalized matching against drug names and active ingredients.
    """
    query_norm = normalize_text(drug_query)
    if not query_norm:
        return None

    brand_family_candidates = _get_brand_family_candidates(df, drug_query)
    query_dosage = extract_dosage_mg(drug_query)
    query_form_hint = _extract_query_form_hint(drug_query)

    exact_mask = (
        df["drug_name_norm"].eq(query_norm)
        | df["brand_name_norm"].eq(query_norm)
        | df["ingredients_norm"].eq(query_norm)
    )
    contains_mask = (
        df["drug_name_norm"].str.contains(query_norm, regex=False)
        | df["brand_name_norm"].str.contains(query_norm, regex=False)
        | df["ingredients_norm"].str.contains(query_norm, regex=False)
        | df["search_text"].str.contains(query_norm, regex=False)
    )

    candidates = df[exact_mask]
    matched_on = "exact_name_or_ingredient"
    if candidates.empty:
        candidates = df[contains_mask]
        matched_on = "partial_name_or_ingredient"

    if candidates.empty and not brand_family_candidates.empty:
        candidates = brand_family_candidates.copy()
        matched_on = "brand_family"

    if candidates.empty:
        query_tokens = [token for token in query_norm.split() if len(token) > 2]
        if query_tokens:
            search_df = brand_family_candidates if not brand_family_candidates.empty else df
            token_mask = pd.Series(False, index=search_df.index)
            for token in query_tokens:
                token_mask = token_mask | search_df["search_text"].str.contains(token, regex=False)
            candidates = search_df[token_mask]
            matched_on = "token_overlap"

    if candidates.empty:
        return None

    best = _select_best_row(candidates, query_norm).copy()
    best["matched_on"] = matched_on
    resolved = _record_to_resolved_dict(best, drug_query=drug_query, country=best["source_country"])

    variant_count = (
        candidates[["atc", "dosage_mg", "form_norm"]]
        .drop_duplicates()
        .shape[0]
    )
    query_has_dosage = extract_dosage_mg(drug_query) is not None
    query_has_form_hint = any(
        token in query_norm.split()
        for token in {"compressa", "compresse", "comprime", "capsula", "capsule", "sciroppo", "supposta", "bustina"}
    )
    ambiguous = variant_count > 1 and not query_has_dosage and not query_has_form_hint

    resolved["ambiguous"] = ambiguous
    resolved["candidate_count"] = int(len(candidates))
    resolved["candidate_variants"] = _candidate_variant_summary(candidates)
    if ambiguous:
        resolved["note"] = "I found multiple forms/strengths, here are the closest matches."
    elif matched_on in {"brand_family", "token_overlap"} and not brand_family_candidates.empty:
        exact_dosage_exists = (
            query_dosage is None
            or candidates["dosage_mg"].apply(lambda value: _coerce_float(value) == query_dosage).any()
        )
        exact_form_exists = (
            not query_form_hint
            or candidates["form_norm"].apply(lambda value: query_form_hint in str(value)).any()
        )
        if not exact_dosage_exists or not exact_form_exists:
            resolved["note"] = "I found this brand in the source country, but not the exact strength/form. Here are the closest matches."

    return resolved


def _get_country_source(registry: dict[str, Any], country: str) -> Any:
    """Return the registered source object for a normalized country key."""
    country_key = normalize_text(country)
    if country_key not in registry:
        raise ValueError(f"Unsupported country: {country}")
    return registry[country_key]


def _is_spain_api_source(source: Any) -> bool:
    """Check whether a registry entry is the live Spain CIMA API adapter."""
    return isinstance(source, dict) and source.get("type") == "cima_api"


def _is_uk_lazy_source(source: Any) -> bool:
    """Check whether a registry entry is the lazy UK dm+d adapter."""
    return isinstance(source, dict) and source.get("type") == "uk_lazy"


def _load_uk_lazy_source(source: dict[str, Any]) -> pd.DataFrame:
    """Load UK dm+d only when a query actually needs UK data."""
    global _UK_DF_CACHE

    if _UK_DF_CACHE is None:
        _UK_DF_CACHE = load_uk_dmd(
            extract_dir=source.get("extract_dir"),
            bonus_dir=source.get("bonus_dir"),
        )
    return _UK_DF_CACHE


def _materialize_country_source(source: Any) -> Any:
    """Return a dataframe/API source, loading lazy local datasets if necessary."""
    if _is_uk_lazy_source(source):
        return _load_uk_lazy_source(source)
    return source


def _load_spain_query_df(drug_query: str, source: dict[str, Any]) -> pd.DataFrame:
    """Load Spain candidates for a free-text query using CIMA."""
    api_base = source.get("api_base", SPAIN_CIMA_API_BASE)
    query_norm = normalize_text(drug_query)
    ingredient_parts = _split_ingredient_names(drug_query)

    df = load_spain_cima(api_base=api_base, nombre=drug_query, max_results=40)
    if not df.empty:
        return df

    practiv1 = ingredient_parts[0] if ingredient_parts else query_norm
    practiv2 = ingredient_parts[1] if len(ingredient_parts) > 1 else None
    df = load_spain_cima(api_base=api_base, practiv1=practiv1, practiv2=practiv2, max_results=40)
    return apply_ingredient_atc_backfill(df, source.get("ingredient_atc_map", {}))


def _load_spain_destination_df(source_resolved: dict[str, Any], source: dict[str, Any]) -> pd.DataFrame:
    """Load Spain destination candidates from CIMA using ATC first, then ingredients."""
    api_base = source.get("api_base", SPAIN_CIMA_API_BASE)
    source_atc = source_resolved.get("atc")
    if source_atc:
        df = load_spain_cima(api_base=api_base, atc=source_atc, max_results=120)
        if not df.empty:
            return apply_ingredient_atc_backfill(df, source.get("ingredient_atc_map", {}))

    ingredient_parts = _split_ingredient_names(source_resolved.get("ingredients"))
    practiv1 = ingredient_parts[0] if ingredient_parts else None
    practiv2 = ingredient_parts[1] if len(ingredient_parts) > 1 else None
    df = load_spain_cima(api_base=api_base, practiv1=practiv1, practiv2=practiv2, max_results=120)
    return apply_ingredient_atc_backfill(df, source.get("ingredient_atc_map", {}))


def _resolve_country_query(country_source: Any, drug_query: str) -> dict[str, Any] | None:
    """Resolve a query against either a dataframe-backed country or a live API-backed country."""
    country_source = _materialize_country_source(country_source)
    if _is_spain_api_source(country_source):
        return resolve_in_country(_load_spain_query_df(drug_query, country_source), drug_query)
    return resolve_in_country(country_source, drug_query)


def match_in_destination(
    source_resolved: dict[str, Any],
    df_dest: pd.DataFrame,
    top_k: int = 10,
) -> dict[str, Any]:
    """Match a resolved source drug against a destination dataset.

    Ranking priority:
    1. ATC match
    2. dosage proximity / equality
    3. form match
    4. ingredient fallback when ATC is missing
    """
    if not source_resolved:
        return {"total_equivalents": 0, "returned_equivalents": 0, "matches": []}

    source_atc = normalize_text(source_resolved.get("atc"))
    source_ingredients = normalize_text(source_resolved.get("ingredients"))
    source_form = _normalize_form(source_resolved.get("form"))
    source_dosage = _coerce_float(source_resolved.get("dosage_mg"))

    if source_atc:
        candidates = df_dest[df_dest["atc"].apply(normalize_text) == source_atc].copy()
        primary_strategy = "atc"
    else:
        candidates = pd.DataFrame()
        primary_strategy = "ingredient_fallback"

    if candidates.empty and source_ingredients:
        ingredient_tokens = [token for token in source_ingredients.split() if len(token) > 2]
        if ingredient_tokens:
            ingredient_mask = pd.Series(False, index=df_dest.index)
            for token in ingredient_tokens:
                ingredient_mask = ingredient_mask | df_dest["ingredients_norm"].str.contains(token, regex=False)
            candidates = df_dest[ingredient_mask].copy()
        else:
            candidates = df_dest[df_dest["ingredients_norm"].eq(source_ingredients)].copy()
        primary_strategy = "ingredient_fallback"

    if candidates.empty:
        return {"total_equivalents": 0, "returned_equivalents": 0, "matches": []}

    def ingredient_overlap(text: str) -> int:
        source_tokens = set(source_ingredients.split())
        dest_tokens = set(str(text).split())
        return len(source_tokens.intersection(dest_tokens))

    scored = candidates.copy()
    scored["atc_match"] = scored["atc"].apply(lambda value: int(normalize_text(value) == source_atc and bool(source_atc)))
    scored["form_match"] = scored["form_norm"].eq(source_form).astype(int)
    scored["ingredient_overlap"] = scored["ingredients_norm"].apply(ingredient_overlap)
    scored["ingredient_match"] = scored["ingredients_norm"].eq(source_ingredients).astype(int)
    scored["dosage_delta"] = scored["dosage_mg"].apply(
        lambda value: abs(_coerce_float(value) - source_dosage)
        if _coerce_float(value) is not None and source_dosage is not None
        else float("inf")
    )
    scored["dosage_match"] = scored["dosage_delta"].apply(lambda delta: int(delta == 0))
    scored["dosage_known"] = scored["dosage_mg"].apply(lambda value: int(_coerce_float(value) is not None))

    scored = scored.sort_values(
        by=[
            "atc_match",
            "dosage_match",
            "form_match",
            "ingredient_match",
            "ingredient_overlap",
            "dosage_known",
            "dosage_delta",
        ],
        ascending=[False, False, False, False, False, False, True],
    )

    total_equivalents = int(len(scored))
    results: list[dict[str, Any]] = []
    for _, row in scored.head(top_k).iterrows():
        notes: list[str] = []
        if not row["atc_match"]:
            notes.append("Exact ATC match not found; matched by ingredient fallback.")
        if source_dosage is not None and not row["dosage_match"]:
            notes.append("Exact dosage match not found.")
        if source_form and not row["form_match"]:
            notes.append("Exact form match not found.")
        if source_ingredients and not row["ingredient_match"]:
            notes.append("Ingredient differs from the resolved source record.")

        results.append(
            {
                "destination_country": row["source_country"],
                "drug_name": row["drug_name"],
                "brand_name": row["brand_name"],
                "ingredients": row["ingredients"],
                "atc": row["atc"] or None,
                "atc_source": row.get("atc_source", "missing"),
                "atc_inference": describe_atc_source(row.get("atc_source", "missing")),
                "dosage_mg": _coerce_float(row["dosage_mg"]),
                "form": row["form"] or None,
                "access_info": row["access_info"] or None,
                "access_summary": summarize_access_info(row["access_info"]),
                "match_strategy": primary_strategy,
                "match_details": {
                    "atc_match": bool(row["atc_match"]),
                    "dosage_match": bool(row["dosage_match"]),
                    "form_match": bool(row["form_match"]),
                    "ingredient_match": bool(row["ingredient_match"]),
                    "ingredient_overlap": int(row["ingredient_overlap"]),
                    "dosage_delta_mg": None
                    if row["dosage_delta"] == float("inf")
                    else float(row["dosage_delta"]),
                },
                "notes": notes,
                "source_record": {
                    "record_id": row["record_id"],
                    "source_country": row["source_country"],
                },
            }
        )

    return {
        "total_equivalents": total_equivalents,
        "returned_equivalents": len(results),
        "matches": results,
    }


def resolve_and_match(
    source_country: str,
    drug_query: str,
    destination_country: str,
    registry: dict[str, Any],
    top_k: int = 10,
) -> dict[str, Any]:
    """Resolve a source-country drug query and find destination-country equivalents."""
    source_key = normalize_text(source_country)
    destination_key = normalize_text(destination_country)

    if source_key not in registry:
        raise ValueError(f"Unsupported source country: {source_country}")
    if destination_key not in registry:
        raise ValueError(f"Unsupported destination country: {destination_country}")

    source_source = _materialize_country_source(registry[source_key])
    destination_source = _materialize_country_source(registry[destination_key])

    resolved = _resolve_country_query(source_source, drug_query)
    if resolved is None:
        return {
            "source_country": source_country,
            "destination_country": destination_country,
            "query": drug_query,
            "resolved_source": None,
            "total_equivalents": 0,
            "returned_equivalents": 0,
            "matches": [],
            "user_summary": build_user_summary(None, [], 0),
        }

    if _is_spain_api_source(destination_source):
        destination_df = _load_spain_destination_df(resolved, destination_source)
    else:
        destination_df = destination_source

    destination_results = match_in_destination(resolved, destination_df, top_k=top_k)
    return {
        "source_country": source_country,
        "destination_country": destination_country,
        "query": drug_query,
        "resolved_source": resolved,
        "total_equivalents": destination_results["total_equivalents"],
        "returned_equivalents": destination_results["returned_equivalents"],
        "matches": destination_results["matches"],
        "note": resolved.get("note"),
        "user_summary": build_user_summary(
            resolved,
            destination_results["matches"],
            destination_results["total_equivalents"],
        ),
    }


def get_registry(force_reload: bool = False) -> dict[str, Any]:
    """Load and cache the country registry for MCP tool calls."""
    global _REGISTRY_CACHE, _UK_DF_CACHE

    if _REGISTRY_CACHE is None or force_reload:
        if force_reload:
            _UK_DF_CACHE = None
        df_it = load_italy_aifa(ITALY_DATA_PATH)
        df_fr = load_france_bdpm(
            FRANCE_DATA_DIR / "CIS_bdpm.txt",
            FRANCE_DATA_DIR / "CIS_COMPO_bdpm.txt",
            FRANCE_DATA_DIR / "CIS_CPD_bdpm.txt",
            FRANCE_DATA_DIR / "CIS_MITM.txt",
        )
        _REGISTRY_CACHE = build_country_registry(
            df_it,
            df_fr,
            spain_config={"type": "cima_api", "api_base": SPAIN_CIMA_API_BASE},
        )
        _REGISTRY_CACHE["uk"] = {"type": "uk_lazy", "extract_dir": None, "bonus_dir": None}

    return _REGISTRY_CACHE


@mcp.tool()
def list_supported_countries() -> dict[str, Any]:
    """Return the countries currently available in the registry."""
    registry = get_registry()
    return {
        "countries": sorted(key.title() for key in registry.keys()),
        "dataset_paths": {
            "Italy": str(ITALY_DATA_PATH),
            "France": str(FRANCE_DATA_DIR),
            "UK": str(_get_uk_extract_dir()),
            "Spain": SPAIN_CIMA_API_BASE,
        },
    }


@mcp.tool()
def reload_datasets() -> dict[str, Any]:
    """Force a reload of the Italy, France and UK datasets into the MCP server cache."""
    registry = get_registry(force_reload=True)
    return {
        "status": "reloaded",
        "countries": sorted(key.title() for key in registry.keys()),
    }


@mcp.tool()
def resolve_drug_in_country(source_country: str, drug_query: str) -> dict[str, Any]:
    """Resolve a drug query in a single country dataset by brand or ingredient."""
    registry = get_registry()
    source_key = normalize_text(source_country)
    if source_key not in registry:
        raise ValueError(f"Unsupported source country: {source_country}")

    resolved = _resolve_country_query(registry[source_key], drug_query)
    return {
        "source_country": source_country,
        "query": drug_query,
        "resolved_source": resolved,
    }


@mcp.tool()
def match_drug_equivalents(
    source_country: str,
    drug_query: str,
    destination_country: str,
    top_k: int = 10,
    ) -> dict[str, Any]:
    """Resolve a source-country drug and return the best destination-country matches."""
    registry = get_registry()
    return resolve_and_match(
        source_country=source_country,
        drug_query=drug_query,
        destination_country=destination_country,
        registry=registry,
        top_k=top_k,
    )


@mcp.tool()
def get_promptopinion_fhir_context_status(ctx: Context) -> dict[str, Any]:
    """Return a safe summary of PromptOpinion FHIR context headers for testing."""
    context = get_promptopinion_fhir_context_from_ctx(ctx)
    return {
        "promptopinion_extension_supported": True,
        "extension_name": PROMPTOPINION_FHIR_EXTENSION_NAME,
        "requested_scopes": get_promptopinion_requested_scopes(),
        "fhir_context_present": any(bool(value) for value in context.values()),
        "fhir_server_url": context.get("fhir_server_url"),
        "patient_id": context.get("patient_id"),
        "has_access_token": bool(context.get("fhir_access_token")),
        "has_refresh_token": bool(context.get("fhir_refresh_token")),
        "has_refresh_url": bool(context.get("fhir_refresh_url")),
    }


def _extract_source_country_from_fhir_extensions(fhir_request: dict[str, Any]) -> str:
    """Extract a source country from a simple or standard FHIR extension shape."""
    extensions = fhir_request.get("extension", [])
    if isinstance(extensions, dict):
        return str(extensions.get("country") or extensions.get("sourceCountry") or "Italy")

    if isinstance(extensions, list):
        for extension in extensions:
            if not isinstance(extension, dict):
                continue
            url = str(extension.get("url", "")).lower()
            if "country" in url:
                return str(extension.get("valueString") or extension.get("valueCode") or "Italy")

    return "Italy"


def _extract_drug_text_from_fhir_request(fhir_request: dict[str, Any]) -> str:
    """Extract medication text from common FHIR MedicationRequest shapes."""
    medication_codeable = fhir_request.get("medicationCodeableConcept")
    if isinstance(medication_codeable, dict) and medication_codeable.get("text"):
        return str(medication_codeable["text"])

    medication = fhir_request.get("medication")
    if isinstance(medication, dict):
        concept = medication.get("concept")
        if isinstance(concept, dict) and concept.get("text"):
            return str(concept["text"])
        if medication.get("text"):
            return str(medication["text"])

    raise KeyError("medication text")


@mcp.tool()
def match_from_fhir(
    fhir_request: dict[str, Any],
    destination_country: str = "UK",
    top_k: int = 5,
) -> dict[str, Any]:
    """Accept a FHIR MedicationRequest-like JSON and return drug equivalents."""
    try:
        drug_text = _extract_drug_text_from_fhir_request(fhir_request)
        source_country = _extract_source_country_from_fhir_extensions(fhir_request)
    except (KeyError, TypeError, AttributeError) as exc:
        return {
            "error": "Invalid FHIR format",
            "detail": str(exc),
            "expected": {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "Tachipirina 500 mg compresse"},
                "extension": {"country": "Italy"},
            },
        }

    return resolve_and_match(
        source_country=source_country,
        drug_query=drug_text,
        destination_country=destination_country,
        registry=get_registry(),
        top_k=top_k,
    )


if __name__ == "__main__":
    mcp.run()
