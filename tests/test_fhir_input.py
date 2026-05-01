import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicine_equivalence_mcp import server as MODULE


class FhirInputTests(unittest.TestCase):
    def test_extracts_drug_from_r4_medication_codeable_concept(self) -> None:
        request = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Tachipirina 500 mg compresse"},
            "extension": {"country": "Italy"},
        }

        self.assertEqual(MODULE._extract_drug_text_from_fhir_request(request), "Tachipirina 500 mg compresse")
        self.assertEqual(MODULE._extract_source_country_from_fhir_extensions(request), "Italy")

    def test_extracts_drug_from_r5_medication_concept(self) -> None:
        request = {
            "resourceType": "MedicationRequest",
            "medication": {"concept": {"text": "Doliprane 500 mg comprime"}},
            "extension": [
                {
                    "url": "https://example.org/fhir/StructureDefinition/source-country",
                    "valueString": "France",
                }
            ],
        }

        self.assertEqual(MODULE._extract_drug_text_from_fhir_request(request), "Doliprane 500 mg comprime")
        self.assertEqual(MODULE._extract_source_country_from_fhir_extensions(request), "France")

    def test_match_from_fhir_calls_existing_matcher(self) -> None:
        request = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Tachipirina 500 mg compresse"},
            "extension": {"country": "Italy"},
        }

        expected = {"matches": [{"drug_name": "DOLIPRANE 500 mg"}]}
        with patch.object(MODULE, "get_registry", return_value={"mock": "registry"}), patch.object(
            MODULE,
            "resolve_and_match",
            return_value=expected,
        ) as mocked_match:
            result = MODULE.match_from_fhir(request, destination_country="France", top_k=3)

        self.assertEqual(result, expected)
        mocked_match.assert_called_once_with(
            source_country="Italy",
            drug_query="Tachipirina 500 mg compresse",
            destination_country="France",
            registry={"mock": "registry"},
            top_k=3,
        )


if __name__ == "__main__":
    unittest.main()
