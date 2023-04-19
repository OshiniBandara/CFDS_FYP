import React from 'react';
import '../App.css';
import { Button } from './Button';
import './HelpSection.css';





function HelpSection() {

  

  return (
    <div className='help-container'>
     
    <div className='help-btns'>
  
        <Button
          onClick={event =>  window.location.href='/detector'}
          className='btns'
          buttonStyle='btn--outline'
          buttonSize='btn--large'>
         GET STARTED
        </Button>
 

        <Button
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'
          onClick={console.log('hey')}
        >
          WATCH DEMO <i className='far fa-play-circle' />
        </Button>
       
    </div>
    </div>
  );
}

export default HelpSection;