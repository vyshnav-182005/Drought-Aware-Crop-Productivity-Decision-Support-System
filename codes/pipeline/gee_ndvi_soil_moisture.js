/*
 * Google Earth Engine script: cropland NDVI + ERA5-Land soil moisture for the 5 districts.
 *
 * How to run:
 *   1. Open https://code.earthengine.google.com (free account; register a non-commercial project).
 *   2. Paste this whole file into a new script and click Run.
 *   3. Check the Console: it must list 5 districts. If fewer, see DISTRICT_NAMES below.
 *   4. Open the Tasks tab and click RUN on both exports. They are saved to your Google Drive
 *      folder "gee_exports" (NDVI takes a few minutes, soil moisture ~10-30 minutes).
 *   5. Download both CSVs into  data/remote_sensing/  in the repo:
 *        ndvi_districts.csv                  (date, district, ndvi, evi)
 *        soil_moisture_era5_districts.csv    (date, district, sm_0_7cm, sm_7_28cm, sm_28_100cm)
 *
 * Values are averaged over each district's polygon (FAO GAUL 2015).
 * NDVI/EVI use cropland pixels only (MODIS land cover classes 12 and 14), 16-day MOD13Q1 at 250 m,
 * pixels with poor quality (SummaryQA > 1) are dropped.
 */

var START = '2000-01-01';
var END = '2026-12-31';

// GAUL spellings differ from DES; every known variant is listed. Output uses canonical names.
var DISTRICT_NAMES = {
  'Ahmadnagar': 'Ahmednagar', 'Ahmednagar': 'Ahmednagar',
  'Bid': 'Beed', 'Beed': 'Beed', 'Bhir': 'Beed',
  'Latur': 'Latur',
  'Osmanabad': 'Osmanabad', 'Dharashiv': 'Osmanabad',
  'Solapur': 'Solapur', 'Sholapur': 'Solapur'
};

var maharashtra = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Maharashtra'));
print('All Maharashtra district names in GAUL (use these if a district is missing):',
      maharashtra.aggregate_array('ADM2_NAME').sort());

var districts = maharashtra
  .filter(ee.Filter.inList('ADM2_NAME', Object.keys(DISTRICT_NAMES)))
  .map(function (f) {
    return f.set('district', ee.Dictionary(DISTRICT_NAMES).get(f.get('ADM2_NAME')));
  });
print('Selected districts (must be 5):', districts.aggregate_array('district'));
Map.centerObject(districts, 7);
Map.addLayer(districts, {color: 'red'}, 'districts');

// ---------------- Cropland mask (MODIS land cover, 2015) ----------------
var lc = ee.ImageCollection('MODIS/061/MCD12Q1')
  .filterDate('2015-01-01', '2015-12-31').first().select('LC_Type1');
var cropMask = lc.eq(12).or(lc.eq(14));
Map.addLayer(cropMask.selfMask().clip(districts), {palette: ['green']}, 'cropland mask', false);

// ---------------- NDVI / EVI (MOD13Q1, 16-day, 250 m) ----------------
var modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(START, END);

var ndviTable = modis.map(function (img) {
  var good = img.select('SummaryQA').lte(1);
  var vi = img.select(['NDVI', 'EVI']).multiply(0.0001)
    .updateMask(good).updateMask(cropMask).rename(['ndvi', 'evi']);
  var date = img.date().format('YYYY-MM-dd');
  return vi.reduceRegions({collection: districts, reducer: ee.Reducer.mean(), scale: 250})
    .map(function (f) {
      return ee.Feature(null, {date: date, district: f.get('district'),
                               ndvi: f.get('ndvi'), evi: f.get('evi')});
    });
}).flatten();

Export.table.toDrive({
  collection: ndviTable, description: 'ndvi_districts', folder: 'gee_exports',
  fileNamePrefix: 'ndvi_districts', fileFormat: 'CSV',
  selectors: ['date', 'district', 'ndvi', 'evi']
});

// ---------------- Soil moisture (ERA5-Land daily, ~11 km) ----------------
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(START, END)
  .select(['volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
           'volumetric_soil_water_layer_3'],
          ['sm_0_7cm', 'sm_7_28cm', 'sm_28_100cm']);

var smTable = era5.map(function (img) {
  var date = img.date().format('YYYY-MM-dd');
  return img.reduceRegions({collection: districts, reducer: ee.Reducer.mean(), scale: 11132})
    .map(function (f) {
      return ee.Feature(null, {date: date, district: f.get('district'),
                               sm_0_7cm: f.get('sm_0_7cm'), sm_7_28cm: f.get('sm_7_28cm'),
                               sm_28_100cm: f.get('sm_28_100cm')});
    });
}).flatten();

Export.table.toDrive({
  collection: smTable, description: 'soil_moisture_era5_districts', folder: 'gee_exports',
  fileNamePrefix: 'soil_moisture_era5_districts', fileFormat: 'CSV',
  selectors: ['date', 'district', 'sm_0_7cm', 'sm_7_28cm', 'sm_28_100cm']
});
