import React, { useState } from 'react';
import '../App.css';
import { Button } from './Button';
import './Upload.css';
//import torch from 'torch';
import axios from "axios";
import { ToastContainer , toast} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';


<ToastContainer
  position="top-right"
  autoClose={5000}
  hideProgressBar={false}
  newestOnTop={false}
  closeOnClick
  rtl={false}
  pauseOnFocusLoss
  draggable
  pauseOnHover
/>





const REPORT = 'http://localhost:3000/Report.txt'
function UploadFile() {

  const [inputFile,setFiles] = useState(null);
  const [datasetName,setDatasetName] = useState('CIFAR10');
 // const [modelPath, setModelPath] = useState('')
  const [output, setOutput] =  useState('')
  const [dataLoaded, setDataLoaded] = useState(false);
  const [loading, setLoading] = useState(false);

  const downloadReport = () => {
    axios.post('http://localhost:5000/download_report', {}, { responseType: 'blob' })
      .then(response => {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'Report.txt');
        document.body.appendChild(link);
        link.click();
      })
      .catch(error => console.error(error));
  };
  
  
  const handleFileSelect = (event) =>{
       setFiles(event.target.files[0]);
  };

  const handleDatasetSelect = (event) =>{
        setDatasetName(event.target.value);
  }; 

   /* const handleFileUpload = async () => {

      if(!inputFile){
        return toast.error('Please select a file!');
      }

       const formData = new FormData();
       formData.append('input_file', inputFile);


       try{
        const response = await axios.post('http://localhost:5000/uploadFile', formData ,{
          headers: {
            'Content-Type' : 'multipart/form-data',
          },
        });
          console.log(response.data);

       } catch(err) {
        console.error(err);
        toast.error('An error occurred while uploading the file!');
  }

  };  */

/*   const handleFileUpload = async () => {
    if (!inputFile) {
      toast.error('No files selected');
      return;
    }
  
    const formData = new FormData();
    formData.append('input_file', inputFile);
  
    try {
      const response = await axios.post('http://localhost:5000/uploadFile', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      toast.success(response.data);
    } catch (err) {
      toast.error(err.message);
    }
  }; */
/* 
  const handleDatasetUpload = async () => {

    if(!datasetFile){
      console.error('No files selected');
      return;
    }

     const formData = new FormData();
     formData.append('dataset_file', datasetFile);


     try{
      const response = await axios.post('http://localhost:5000/uploadDataset', formData ,{
        headers: {
          'Content-Type' : 'multipart/form-data',
        },
      });
        console.log(response.data);

     } catch(err) {
      console.error(err);
     }

}; */





const handleSubmit = (event) => {

  setLoading(true);
  event.preventDefault();

  if (!inputFile || !datasetName) {
    setLoading(false);
    toast.error('Please select a file and a dataset!');
    return;
  }

    const formData = new FormData();
    formData.append('model_file', inputFile);
    formData.append('dataset_name', datasetName);
    axios.post('http://localhost:5000/train', formData)
      .then((response) => {
      //setTrainedLoss(response.data.lose);
      //setTrainedAccuracy(response.data.accuracy);
      console.log(response.data.output)
      setOutput(response.data.output);
      setLoading(false);
      setDataLoaded(true);
  })
     .catch((error) => {
      toast.error('An error occurred while training the model!');
      setLoading(false);
      
    });

  };

const datasetOptions = [
  
    'CIFAR10',
    'CIFAR100',
    
  ];



  return (
    <div className='up'>
      <h1>Please Input Your Machine Learning Model and Dataset</h1>
       
        {/* <h3>{ status } </h3> */}

       
      
        {/* <div>
           <input type="file" name="model_file" onChange={handleFileSelect}/>
           <button onClick={handleFileUpload}>Upload Model</button>
         </div> */}
        <div>
          <form onSubmit={handleSubmit}>
          <div className='up-container2'>
          <label>Upload PyTorch Model</label>
            <input className='filepicker' type="file" name="model_file" onChange={handleFileSelect} />
          </div>
            <div className='up-container3'>
              <label>Select PyTorch Dataset</label>
              <select className='selectvalue' value={datasetName} onChange={handleDatasetSelect}> 
                {datasetOptions.map((datasetOption) => (
                  <option key={datasetOption} value={datasetOption}>
                    {datasetOption}
                  </option>
                ))}
              </select>
              <ToastContainer/>
            </div> 

        <br />
        
        <div className='up-container4'>
          <br></br>
        <button> 
          Train Model
          </button>
          <ToastContainer/>
         </div> 
          <div className='up-container6'>
          {loading && <div className='loader'>  </div> }
          <br></br><br></br>
         
          {loading && <div><h5>Loading...</h5></div>}
          </div>
      
      </form>
    <div>
    {dataLoaded ? (
     <p>The model accuracy differences according to classes are : {output.toString()}</p>
     ) : (
              
      <p> </p>
    )}
     </div>
     </div> 
  
      <br></br>
      <br></br>
      <br></br>
      <div className='up-container5'>
            {dataLoaded ? (
            <h4>Catastrophic Forgetting Detected!</h4>
            ) : (
              
              <p>  </p>
            )}
      </div>


    <div className='up-container'>
     
     <div className='up-btns'>
     {dataLoaded ? ( 
       
      <Button
          onClick={downloadReport}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
        Download Test Report
        </Button>
      ) : (
              
      <p>  </p>
    )}
      
    </div>
    </div>
  
    </div>
        );
    }
  
    
    export default UploadFile;