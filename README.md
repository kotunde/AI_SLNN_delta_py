# Egyrétegű neuronháló - delta szabály

Használt programozási nyelv: Python

### Megoldás

#### Előfeldolgozás
- `load_data` 
  - képek betöltése a folderekből
  - képek paddingolása egységesítésért
- `preprocess_image`
  - képek átméretezése, színcsatorna redukálása, álló képek elforgatása, képmátrix sorvektorrá konvertálása
- `split_data`
  - képek betöltése a két megadott folderből
  - képek egyesítése, felosztása tanító és teszthalmazokra
- `offlineLearning`
  - egyrétegű neruonháló tanítása a bemeneti adat alapján négyzetes hibafüggvényt használva
  - tanítási ciklus kilépési feltételei: iterációszám, hiba változásának mértéke
- `predict_test_data`
  - a teszthalmaz becslése adott súlyzók segítségével
- `check_prediciton`
  - a becsült és valódi címkék összehasonlítása
- `calc_score`
  - a háló teljesítményének kiszámítása a becsült és valódi értékek alapján
- `plot_test_image`
  - a teszthalmaz egy részének (fele) megjelenítése a becsült címkékkel
- `plot_confusion_matrix`
  - konfúziós mátrix megjelenítése címkékkel


### Problémák
- kicsi az adathalmaz (--> tanítási-tesztelési arány növelése)
- az pixelértékek normalizálása/centralizálása 
