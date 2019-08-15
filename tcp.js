// rand function
function randint(min, max){
	return Math.floor(Math.random() * (max-min)) + min;
}

// get a random color
function getColor(){
	var R = randint(0, 256) / 256;
	var G = randint(0, 256) / 256;
	var B = randint(0, 256) / 256;
	var color = [R, G, B];

	return color;
}

// update color of the block
function updateCol(color = false){
	var colBlock = document.getElementById("color");
	if(color == false){
		color = getColor();
	}
	var rgb = "rgb(" + color[0] * 256 + ", " + color[1] * 256 + ", " + color[2] * 256 + ")";
	colBlock.style.backgroundColor = rgb;
}

function updateText(res){
	if(res > 0.5){
		document.getElementById("ctext").style.color = "#ffffff";
	}
	else{
		document.getElementById("ctext").style.color = "#000000";
	}
}

// decide if the text in it should be black or white
function findTextCol(color){
	var threshold = 128 / 256; //threshold could be anything
	var target = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114;

	if(target > threshold){
		return 0; //text color should be BLACK
	}else{
		return 1; //text color should be WHITE
	}
}

function funcOnMat(func, matrix){
	var newMatrix = [];
	for(let i = 0; i<matrix.length; i++){
		let line = [];
		for(let j = 0; j<matrix[i].length; j++){
			line.push(func(matrix[i][j]));
		}
		newMatrix.push(line);
	}
	return newMatrix;
}

function funcOnArr(func, arr){
	var newArr = [];
	for(let i = 0; i<arr.length; i++){
		newArr.push(func(arr[i]));
	}
	return newArr;
}

// activation functions
function sigmoid(x){
	return 1 / (1 + Math.exp(-x));
}
// derivative of sigmoid
function sigmoid_p(x){
	return sigmoid(x) * (1 - sigmoid(x));
}

function newRandMatrix(lines, columns){
	var matrix = [];
	for(let i = 0; i<lines; i++){
		let line = [];
		for(let j = 0; j<columns; j++){
			line.push(Math.random());
		}
		matrix.push(line);
	}
	return matrix;
}

function newRandArr(lines){
	var arr = [];
	for(let i = 0; i<lines; i++){
		arr.push(Math.random());
	}
	return arr;
}

function lineMultiply(m1, m2){
	var newArr = [];
	for(let i = 0; i<m1.length; i++){
		newArr.push(m1[i] * m2[i]);
	}
	return newArr;
}

function updateW(w, dcostw){
	for(let i=0; i<dcostw.length; i++){
		for(let j=0; j<dcostw[i].length; j++){
			dcostw[i][j] = 0 - learn_rate * dcostw[i][j];
		}
	}
	return math.add(w, dcostw);
}

function updateB(b, dcostb){
	for(let i=0; i<dcostb.length; i++){
		dcostb[i] = 0 - learn_rate * dcostb[i];
	}
	return math.add(b, dcostb);
}

var nodes_in = 3;
var nodes_hidden = 5;
var nodes_out = 1;

var learn_rate = 0.2;

// initialising weights and biases
var wH = newRandMatrix(nodes_in, nodes_hidden);
var bH = newRandArr(nodes_hidden);

var wOut = newRandMatrix(nodes_hidden, nodes_out);
var bOut = newRandArr(nodes_out);

function train(){
	document.getElementById("instruction").innerHTML = "training... please give it some time";
	document.getElementById("trainb").onclick = null;
	document.getElementById("testb").onclick = null;
	setTimeout(net, 2);
}


function net(){
	
	var zH, predH, zOut, pred, cost;
	var dcost_pred, dpred_zOut, dzOut_wOUt, dzOut_bOut, dcost_wOut, dcost_bOut;
	var dcost_zOut, dzOut_predH, dpredH_zH, dzH_wH, dzH_bH, dcost_predH, dcost_zH, dcost_wH, dcost_bH;
	
	for(let i = 0; i<50000; i++){
		var color = getColor();
		var target = findTextCol(color);

		// FEED FORWARD
		zH = math.multiply(color, wH);
		zH = math.add(zH, bH);
		predH = funcOnArr(sigmoid, zH);

		zOut = math.multiply(predH, wOut);
		zOut = math.add(zOut, bOut);
		pred = funcOnArr(sigmoid, zOut);

		// BACK PROPAGATION
		// out layer to hidden layer
		cost = (pred - target) ** 2;

		dcost_pred = 2 * (pred - target);
		dpred_zOut = sigmoid_p(zOut);
		dzOut_wOut = predH;
		dzOut_bOut = 1;

		dcost_wOut = dcost_pred * dpred_zOut;
		dcost_wOut = math.multiply(math.transpose(dzOut_wOut), dcost_wOut);
		dcost_wOut = math.transpose([dcost_wOut]);
		dcost_bOut = dcost_pred * dpred_zOut * dzOut_bOut;

		// hidden layer to input layer
		dcost_zOut = dcost_pred * dpred_zOut;

		dzOut_predH = wOut;
		dpredH_zH = funcOnArr(sigmoid_p, zH);
		dzH_wH = color;
		dzH_bH = 1;

		dcost_predH = math.multiply(dcost_zOut, math.transpose(dzOut_predH));
		dcost_zH = lineMultiply(dcost_predH[0], dpredH_zH);

		dcost_wH = math.multiply(math.transpose([dzH_wH]), [dcost_zH]);
		dcost_bH = math.multiply(dcost_zH, dzH_bH);

		// UPDATE WEIGHTS AND BIAS
		wOut = math.subtract(wOut, math.multiply(dcost_wOut, learn_rate))
		wH = math.subtract(wH, math.multiply(dcost_wH, learn_rate))

		bOut = math.subtract(bOut, math.multiply(dcost_bOut, learn_rate))
		bH = math.subtract(bH, math.multiply(dcost_bH, learn_rate))
	}
	done();
}

function train2(){
	document.getElementById("instruction").innerHTML = "no reason to train more :)"
}

function done(){
	document.getElementById("instruction").innerHTML = "done! click on test to try a random color";
	document.getElementById("trainb").onclick = train2;
	document.getElementById("testb").onclick = test;
}

function showHelp(){
	let hlp = document.getElementById("about");
	if(window.getComputedStyle(hlp).getPropertyValue("display") == "none"){
		hlp.style.display = "block";
	}
	else{
		hlp.style.display = "none";
	}
}

function test(){
	color = getColor();
	let target = findTextCol(color);

	// one last feed forward :)
	let zH = math.multiply(color, wH);
		zH = math.add(zH, bH);
	let predH = funcOnArr(sigmoid, zH);

	let zOut = math.multiply(predH, wOut);
		zOut = math.add(zOut, bOut);
	let pred = funcOnArr(sigmoid, zOut);

	updateCol(color);
	updateText(pred);
	var msg = "predicted value was " + parseFloat(pred).toFixed(9);
	msg = msg + "; target value was " + target;
	document.getElementById("instruction").innerHTML = msg;

}

function init(){
	updateCol();
}
