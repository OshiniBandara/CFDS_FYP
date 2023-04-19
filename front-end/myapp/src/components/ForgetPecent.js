import React from 'react';
import '../App.css';
import { Button } from './Button';
import './ForgetPercent.css';




function ForgetPecent() {

  

    return (
     <div className='fg'>
        <h1>Self-Supervised Learning Signal to Overcome Forgetfulness of the Model </h1>
        

         <h3>OVERCOME FORGETFULNESS</h3>

    <div className='fg-container'>
     
     <div className='fg-btns'>
     
      <Button
          onClick={event =>  window.location.href='/test'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            Start
          </Button>
          
     
    </div>
    </div>

    </div> 

    
        );
    }
    
    export default ForgetPecent;