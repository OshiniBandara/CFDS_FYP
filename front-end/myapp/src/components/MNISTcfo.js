import React, { useState , useEffect } from 'react';
import '../App.css';
import './Detection.css';
import axios from 'axios';
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

function CIFAR10cfd() {

  const [output1, setOutput1] =  useState('');
  const [output2, setOutput2] =  useState('');
 
  const [dataLoaded1, setDataLoaded1] = useState(false);
  const [dataLoaded2, setDataLoaded2] = useState(false);
  const [loading1, setLoading1] = useState(false);
  const [loading2, setLoading2] = useState(false);

    
    
      

  const handleClick = () => {
    setLoading1(true);
    axios.get('http://localhost:5000/normal3').then (response => {
       // Parse the JSON response and access the first element
      const outputString = response.data.output[0];

      // Remove the double quotes from the string
       const formattedOutput = outputString.replace(/^"(.*)"$/, '$1');
       console.log(formattedOutput);
       setOutput1(formattedOutput);
       setLoading1(false);
       setDataLoaded1(true);
      
     })
     
     .catch(error => {
       console.error(error);
       toast.error('An error occurred while training the model!');
       setLoading1(false);
     }
     );

 }

 const handleClick2 = () => {
   setLoading2(true);
   axios.get('http://localhost:5000/ssls3').then (response => {
      // Parse the JSON response and access the first element
      const outputString = response.data.output[0];

      // Remove the double quotes from the string
      const formattedOutput = outputString.replace(/^"(.*)"$/, '$1');

      console.log(formattedOutput);
      setOutput2(formattedOutput);
      setLoading2(false);
      setDataLoaded2(true);
     
    })
    
    .catch(error => {
      console.error(error);
      toast.error('An error occurred while training the model!');
      setLoading2(false);
    }
    );

}
    
      return(
        <div>
           <div className='dt-container'>
          
          <button className='btndt' onClick={handleClick}> Train MNIST Dataset without SSLS</button>
          <ToastContainer/>
           {loading1 && <div className='loader'></div>}
           {loading1 && <div><h5>Loading...</h5></div>}
         
          </div>
          <div className='dt2-container'>
          <h2>Accuracy for MNIST Dataset without SSLS</h2><br/>
          
          {dataLoaded1 && <p>{output1}</p>}
            </div>

          <div className='dt-container'>
          
          <button className='btndt' onClick={handleClick2}> Train MNIST Dataset with SSLS </button>
          <ToastContainer/>
           {loading2 && <div className='loader'></div>}
           {loading2 && <div><h5>Loading...</h5></div>}
         
          </div>
          <div className='dt2-container'>
          <h2>Accuracy for MNIST Dataset with SSLS</h2><br/>
          
          {dataLoaded1 && <p>{output2}</p>}
            </div>
          


          
          <div className='btn-container'>
            <button className='btn'
            onClick={event =>  window.location.href='/experiments'}>
             Back
            </button>
            </div>


            </div>
       
        
        
        ); 



    }

    
    export default CIFAR10cfd;