import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicine_equivalence_mcp import server as MODULE


def make_country_df(rows: list[dict]) -> pd.DataFrame:
    """Build a normalized dataframe compatible with the matcher."""
    df = pd.DataFrame(rows).fillna("")
    if "dosage_mg" not in df.columns:
        df["dosage_mg"] = df["drug_name"].apply(MODULE.extract_dosage_mg)
    df["form_norm"] = df["form"].apply(MODULE._normalize_form)
    df["drug_name_norm"] = df["drug_name"].apply(MODULE.normalize_text)
    df["brand_name_norm"] = df["brand_name"].apply(MODULE.normalize_text)
    df["ingredients_norm"] = df["ingredients"].apply(MODULE.normalize_text)
    if "atc_source" not in df.columns:
        df["atc_source"] = df["atc"].apply(lambda value: "direct_dataset" if str(value).strip() else "missing")
    df["search_text"] = df.apply(
        lambda row: MODULE._build_search_text(row["drug_name"], row["brand_name"], row["ingredients"]),
        axis=1,
    )
    return df


class MedicineEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df_it = make_country_df(
            [
                {
                    "record_id": "it1",
                    "source_country": "Italy",
                    "drug_name": "Tachipirina 500 mg compresse",
                    "brand_name": "Tachipirina",
                    "ingredients": "Paracetamolo",
                    "atc": "N02BE01",
                    "form": "Compressa",
                    "access_info": "OTC",
                },
                {
                    "record_id": "it2",
                    "source_country": "Italy",
                    "drug_name": "Augmentin 875 mg + 125 mg compresse rivestite con film",
                    "brand_name": "Augmentin",
                    "ingredients": "Amoxicillina + Acido Clavulanico",
                    "atc": "J01CR02",
                    "form": "Compressa rivestita con film",
                    "access_info": "Ricetta",
                },
            ]
        )

        self.df_fr = make_country_df(
            [
                {
                    "record_id": "fr1",
                    "source_country": "France",
                    "drug_name": "DOLIPRANE 500 mg, comprimé",
                    "brand_name": "DOLIPRANE",
                    "ingredients": "Paracétamol",
                    "atc": "N02BE01",
                    "form": "comprimé",
                    "access_info": "Non soumis à prescription",
                },
                {
                    "record_id": "fr2",
                    "source_country": "France",
                    "drug_name": "PARACETAMOL ARROW 500 mg, comprimé",
                    "brand_name": "PARACETAMOL ARROW",
                    "ingredients": "Paracétamol",
                    "atc": "N02BE01",
                    "form": "comprimé",
                    "access_info": "Non soumis à prescription",
                },
                {
                    "record_id": "fr3",
                    "source_country": "France",
                    "drug_name": "Augmentin 1 g/125 mg, poudre pour suspension buvable en sachet-dose",
                    "brand_name": "Augmentin",
                    "ingredients": "Amoxicilline + Acide clavulanique",
                    "atc": "J01CR02",
                    "form": "poudre pour suspension buvable",
                    "access_info": "Prescription obligatoire",
                },
            ]
        )

        self.df_uk = make_country_df(
            [
                {
                    "record_id": "uk1",
                    "source_country": "UK",
                    "drug_name": "Calpol 500mg tablets",
                    "brand_name": "Calpol",
                    "ingredients": "Paracetamol",
                    "atc": "N02BE01",
                    "atc_source": "bonus_bnf",
                    "form": "Tablet",
                    "access_info": "P",
                },
                {
                    "record_id": "uk2",
                    "source_country": "UK",
                    "drug_name": "Co-amoxiclav 625mg tablets",
                    "brand_name": "Co-amoxiclav",
                    "ingredients": "Amoxicillin + Clavulanic acid",
                    "atc": "J01CR02",
                    "atc_source": "bonus_bnf",
                    "form": "Tablet",
                    "access_info": "POM",
                },
            ]
        )

        self.df_who = pd.DataFrame(
            [
                {
                    "atc_code": "N02BE01",
                    "atc_name": "paracetamol",
                    "ddd": "3",
                    "uom": "g",
                    "adm_r": "O",
                    "note": "",
                },
                {
                    "atc_code": "J01CA04",
                    "atc_name": "amoxicillin",
                    "ddd": "1.5",
                    "uom": "g",
                    "adm_r": "O",
                    "note": "",
                },
                {
                    "atc_code": "J01CR02",
                    "atc_name": "combinations of penicillins, incl. beta-lactamase inhibitors",
                    "ddd": "NA",
                    "uom": "NA",
                    "adm_r": "NA",
                    "note": "",
                },
            ]
        )

        self.registry = MODULE.build_country_registry(
            self.df_it,
            self.df_fr,
            df_uk=self.df_uk,
            df_who=self.df_who,
            spain_config={"type": "cima_api", "api_base": MODULE.SPAIN_CIMA_API_BASE},
        )

    def test_italy_to_france_tablet_match(self) -> None:
        result = MODULE.resolve_and_match("Italy", "Tachipirina 500 mg tablets", "France", self.registry, top_k=5)

        self.assertIsNotNone(result["resolved_source"])
        self.assertEqual(result["resolved_source"]["atc"], "N02BE01")
        self.assertGreaterEqual(result["total_equivalents"], 1)
        self.assertEqual(result["matches"][0]["brand_name"], "DOLIPRANE")
        self.assertTrue(result["matches"][0]["match_details"]["atc_match"])
        self.assertTrue(result["matches"][0]["match_details"]["form_match"])

    def test_france_to_uk_match(self) -> None:
        result = MODULE.resolve_and_match("France", "Doliprane 500 mg comprimé", "UK", self.registry, top_k=5)

        self.assertEqual(result["resolved_source"]["brand_name"], "DOLIPRANE")
        self.assertEqual(result["resolved_source"]["atc"], "N02BE01")
        self.assertEqual(result["matches"][0]["brand_name"], "Calpol")
        self.assertEqual(result["matches"][0]["atc_source"], "bonus_bnf")

    def test_uk_to_italy_match(self) -> None:
        result = MODULE.resolve_and_match("UK", "Calpol 500mg tablets", "Italy", self.registry, top_k=5)

        self.assertEqual(result["resolved_source"]["brand_name"], "Calpol")
        self.assertEqual(result["resolved_source"]["atc_source"], "bonus_bnf")
        self.assertEqual(result["matches"][0]["brand_name"], "Tachipirina")
        self.assertEqual(result["matches"][0]["atc"], "N02BE01")

    def test_who_fallback_only_for_single_ingredient(self) -> None:
        df_fr_missing = make_country_df(
            [
                {
                    "record_id": "fr_missing_single",
                    "source_country": "France",
                    "drug_name": "DOLIPRANE 500 mg, comprimé",
                    "brand_name": "DOLIPRANE",
                    "ingredients": "Paracétamol",
                    "atc": "",
                    "form": "comprimé",
                    "access_info": "Non soumis à prescription",
                },
                {
                    "record_id": "fr_missing_combo",
                    "source_country": "France",
                    "drug_name": "AMOXICILLINE/ACIDE CLAVULANIQUE 500 mg/62,5 mg, comprimé",
                    "brand_name": "AMOXICILLINE/ACIDE CLAVULANIQUE",
                    "ingredients": "Amoxicilline + Acide clavulanique",
                    "atc": "",
                    "form": "comprimé",
                    "access_info": "Prescription obligatoire",
                },
            ]
        )

        registry = MODULE.build_country_registry(
            self.df_it,
            df_fr_missing,
            df_uk=self.df_uk,
            df_who=self.df_who,
            spain_config={"type": "cima_api", "api_base": MODULE.SPAIN_CIMA_API_BASE},
        )

        france_df = registry["france"]
        single_row = france_df.loc[france_df["record_id"] == "fr_missing_single"].iloc[0]
        combo_row = france_df.loc[france_df["record_id"] == "fr_missing_combo"].iloc[0]

        self.assertEqual(single_row["atc"], "N02BE01")
        self.assertEqual(single_row["atc_source"], "who_atc_ingredient")
        self.assertEqual(combo_row["atc"], "")
        self.assertEqual(combo_row["atc_source"], "missing")


if __name__ == "__main__":
    unittest.main()
