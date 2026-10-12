# Historical surface-data availability, 12 October

Read-only source discovery found no immediately usable additional archive.
No provider was contacted, no account was created, and no data was imported.

- [FlyItalyADSB's archive documentation](https://flyitalyadsb.com/en/dati/archivio/)
  limits self-service history to roughly 18–20 recent days. Older storage needs
  a separate request. The stated distributed-data licence is CC BY-SA 4.0;
  the documentation does not establish January/July 2026 availability.
- [adsb.aero's API guide](https://adsb.aero/llms.txt) identifies its source as
  ADSB.lol and explicitly excludes ground points. It does not supply the
  missing surface observations needed here.
- [Airplanes.live's published API](https://airplanes.live/api-docs/) documents
  live and last-known positions, without a historical bulk-data endpoint.
  Its [FAQ](https://airplanes.live/faq/) describes replay access for feeders;
  neither page establishes a self-service research archive for our dates.
- [ADSB Exchange's samples](https://adsbexchange.com/data-products/sample-data/)
  cover the first day of each month; they are described as evaluation samples.
  That page alone does not establish the open-data reuse permission needed
  for a new competition input.

The newly published [reliable-hamburger repository](https://github.com/MariusEtudiant/prc-data-challenge-2026)
reports 282.40 seconds and describes copy mixtures, plan bounds and ADS-B
pushback features. Those broad methods are already represented in our retained
work; its reported score is not an independent evaluation by this project.
No third-party code was copied or executed.
