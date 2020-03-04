var chemcalc = require('chemcalc');
var result = chemcalc.mfFromMonoisotopicMass(1301.314833,{'mfRange':'C1-200H1-200N0-5O0-10S0-10','maxUnsaturation':'10','useUnsaturation':'true','integerUnsaturation':'true','massRange':'0.005'});
console.log(result)

