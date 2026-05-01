# Dataset Setup

This folder is intentionally lightweight and safe to publish.

It does **not** include the large raw datasets used by MedMap.
Instead, it contains:

- official dataset source links
- optional convenience bundle information
- setup notes for people who want to run the project locally

## Official Sources

- Italy (AIFA): https://www.aifa.gov.it/web/guest/liste-dei-farmaci
- France (BDPM): https://base-donnees-publique.medicaments.gouv.fr/telechargement
- UK (NHS dm+d / TRUD): https://isd.digital.nhs.uk/trud/user/guest/group/0/pack/6/subpack/24/releases
- Spain (AEMPS / CIMA): https://sede.aemps.gob.es/datos-abiertos/

## Optional Convenience Bundle

If you want a pre-bundled archive for deployment or testing, you can use:

- https://pub-bfd515f1bc1a4c079a3db731bf46d3d4.r2.dev/Datasets.zip

This bundle is a convenience copy, not the official upstream source.

## Expected Layout

When extracted, the dataset structure should provide country folders such as:

```text
Datasets/
  Italy/
  France/
  UK/
```

Spain is loaded through the public CIMA API rather than a local dataset folder.
