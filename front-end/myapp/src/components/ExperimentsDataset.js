import React from 'react';
import '../App.css';
import { Button } from './Button';
import './Test.css';



function ExperimentsDataset() {

 

 
  

    return (
     <div className='fg'>
        <h3>Experiments for Catastrophic Forgetting Detection</h3>
        

         <h1>Select The Dataset You Want To Test</h1>
        

    <div className='fg-container'>
     
     <div className='fg-btns'>
     

      <Button
          onClick={event =>  window.location.href='/CfdCifar10'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            CIFAR 10
          </Button>

          <Button
          onClick={event =>  window.location.href='/CfdCifar100'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
           CIFAR 100
          </Button>

          <Button
          onClick={event =>  window.location.href='/CfdMNIST'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            MNIST
          </Button>

          <Button
          onClick={event =>  window.location.href='/CfdSVHN'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            SVHN
          </Button>
          
     
    </div>

    </div>

    
    <div className='fg-btns'>
     
     </div>

 
        <h3>Experiments for Overcoming Catastrophic Forgetting</h3>
        

         <h1>Select The Dataset You Want To Overcome Catastrophic Forgetting</h1>

        

    <div className='fg-container'>
     
     <div className='fg-btns'>
     

     <Button
          onClick={event =>  window.location.href='/CfoCifar10'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            CIFAR 10
          </Button>

          <Button
          onClick={event =>  window.location.href='/CfoCifar100'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
           CIFAR 100
          </Button>

          <Button
          onClick={event =>  window.location.href='/CfoMNIST'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            MNIST
          </Button>

          <Button
          onClick={event =>  window.location.href='/CfoSVHN'}
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'>
            SVHN
          </Button>
         
          </div>

    </div>

  </div>




    
        );
    }
    
    export default ExperimentsDataset;