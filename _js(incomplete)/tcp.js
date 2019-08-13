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
	console.log(res)
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
	return 1 / (1 + Math.exp(x));
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

var loops = 0;

function train(){
	var trn = setInterval(ffbp, 0.5);
	var check = setInterval(checktr, 0.5);
	document.getElementById("ctext").innerHTML = "training...";

	function checktr(){
		document.getElementById("ctext").innerHTML = bOut;
		if(loops > 10000){
			clearInterval(trn);
			document.getElementById("ctext").innerHTML = "done!";
			loops = 0;
			clearInterval(checktr);
		}
	}
}

function ffbp(){
	loops++;
	let color = getColor();
	let target = findTextCol(color);

	// FEED FORWARD
	let zH = math.multiply(color, wH);
		zH = math.add(zH, bH);
	let predH = funcOnArr(sigmoid, zH);

	let zOut = math.multiply(predH, wOut);
		zOut = math.add(zOut, bOut);
	let pred = funcOnArr(sigmoid, zOut);

	// BACK PROPAGATION
	// out layer to hidden layer
	let cost = (pred - target) ** 2;

	let dcost_pred = 2 * (pred - target);
	let dpred_zOut = sigmoid_p(zOut);
	let dzOut_wOut = predH;
	let dzOut_bOut = 1;

	let dcost_wOut = dcost_pred * dpred_zOut;
		dcost_wOut = math.multiply(math.transpose(dzOut_wOut), dcost_wOut);
		dcost_wOut = math.transpose([dcost_wOut]);
	let dcost_bOut = dcost_pred * dpred_zOut * dzOut_bOut;

	// hidden layer to input layer
	let dcost_zOut = dcost_pred * dpred_zOut;

	let dzOut_predH = wOut;
	let dpredH_zH = funcOnArr(sigmoid_p, zH);
	let dzH_wH = color;
	let dzH_bH = 1;

	let dcost_predH = math.multiply(dcost_zOut, math.transpose(dzOut_predH));
	let dcost_zH = lineMultiply(dcost_predH[0], dpredH_zH);

	let dcost_wH = math.multiply(math.transpose([dzH_wH]), [dcost_zH]);
	let dcost_bH = math.multiply(dcost_zH, dzH_bH);

	// UPDATE WEIGHTS AND BIAS
	wOut = updateW(wOut, dcost_wOut);
	wH = updateW(wH, dcost_wH);

	bOut = updateB(bOut, dcost_bOut);
	bH = updateB(bH, dcost_bH);

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

}

function init(){
	updateCol();
}
