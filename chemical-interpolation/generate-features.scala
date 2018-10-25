/**
 * Read in dataset, generate physical features and atomic fraction
 */

import java.io.File;
import magpie.data.materials.CompositionDataset;
import magpie.attributes.generators.composition.ElementFractionAttributeGenerator;

val datafile = args(0);
val sampleName = datafile.split("\\.")(0);

// Load in dataset
val data = new CompositionDataset();
data.importText(datafile, null);
println(s"Read in ${data.NEntries()} from $datafile");

// Set formation enthalpy as property to be modelled
data.setTargetProperty("delta_e", true);
println(s"Set ${data.getTargetPropertyName()} as property to be modeled");

// Generate attributes
data.addElementalPropertySet("general");
data.setDataDirectory("../magpie/lookup-data");
data.generateAttributes();
println(s"Generated ${data.NAttributes()} attributes");

// Save them to disk 
data.saveCommand(s"${sampleName}-physical", "csv");

// Generate element fractions
data.clearAttributeGenerators();
data.addAttributeGenerator(new ElementFractionAttributeGenerator());
data.generateAttributes();

data.saveCommand(s"${sampleName}-fractions", "csv");