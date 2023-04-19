import React from 'react';
import './Cards.css';
import CardItem from './CardItem';


function Cards() {
  return (
    <div className='cards'>
      <h1>Start Detecting Forgetfulness of Your Model</h1>
      <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            
            <CardItem
              src='./assets/images/upload.jpeg'
              label='Step 01'
              text='Upload your Machine Learning Model and Dataset'
              path='/detector'
            />
            <CardItem
              src='./assets/images/Train.jpeg'
              label='Step 02'
              text='Train your Model and Get Forgetting Percentage'
              path='/detector'
            />
            
            </ul>
        </div>
      </div>
    </div>
  );
}

export default Cards;