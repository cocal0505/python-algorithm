const spawn = require('child_process').spawn
const  array = [[255, 0, 0], [244, 247, 114], [252, 255, 248], [186, 205, 219], [-999, -999, -999], [244, 247, 114], [244, 247, 114], [186, 205, 219], [-999, -999, -999], [244, 247, 114], [-999, -999, -999]]
const process1 = spawn('python', ['./repeat.py',array]);


const datafrompython = []

process1.stdout.on('data',data=>{
    
    // console.log(data.toString())
    datafrompython.push(data.toString());
    console.log("from python ",datafrompython)
})





