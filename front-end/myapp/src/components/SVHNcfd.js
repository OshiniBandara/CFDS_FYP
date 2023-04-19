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



function SVHNcfd() {

      const [output, setOutput] =  useState([''])
      const [dataLoaded, setDataLoaded] = useState(false);
      const [loading, setLoading] = useState(false);
    
       const [rows, setRows] = useState([]);

        // update the rows state when the output prop changes
        useEffect(() => {
          // create a new row for each group of 5 items in the output
          const newRows = [];
          for (let i = 0; i < output.length; i += 5) {
            const row = output.slice(i, i + 5);
            newRows.push(row);
          }

          // update the rows state with the new rows
          setRows((prevRows) => [...prevRows, ...newRows]);
        }, [output]);
      

      const handleClick = () => {
         setLoading(true);
         axios.get('http://localhost:5000/run4').then (response => {
          
            console.log(response.data);
            setOutput(response.data.output);
            setLoading(false);
            setDataLoaded(true);
           
          })
          
          .catch(error => {
            console.error(error);
            toast.error('An error occurred while training the model!');
            setLoading(false);
          }
          );

      }
    
      return(
        <div>
           <div className='dt-container'>
          
          <button className='btndt' onClick={handleClick}> Train SVHN Dataset </button>
          <ToastContainer/>
           {loading && <div className='loader'></div>}

           <br></br><br></br>
         
         {loading && <div><h5>Loading...</h5></div>}
         
          </div>
          <div className='dt2-container'>
          <h2>The SVHN dataset will be split into two parts and
             trained it. <br/> After training the first part it will test for the test dataset and 
             gather the class vice accuracies for the test dataset.  <br/> This method repeats for
              the second part of the dataset and after gathering the class vice accuracies, 
              <br/> here is the result of forgetting percentages of test classes.
           </h2>
            </div><br/>
          {dataLoaded ? (
            <div className='dt5-container'>
            
            <div className='output-container'>
          <table>
              <tbody>
                {rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>




                </div>
              
              
            </div>
            ):(
              <br/>
             )}

          
    


          <div className='dt3-container'>
            {dataLoaded ? (
            <h3>Catastrophic Forgetting Detected!</h3>
            ) : (
              <h1> Try Test for SVHN Dataset </h1>
            )}
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
    
    export default SVHNcfd;