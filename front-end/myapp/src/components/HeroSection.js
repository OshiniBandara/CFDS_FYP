import React from 'react';
import '../App.css';
import { Button } from './Button';
import './HeroSection.css';
import { useNavigate } from "react-router-dom";


function HeroSection() {

  const navigate = useNavigate();

  const openInNewTab = (url) => {
    window.open(url, '_blank', 'noreferrer');
  };

  return (
    <div className='hero-container'>
      {/* <video src='/src/assets/videos/video-1.mp4' autoPlay loop muted />  */}
      <h1>ADVENTURE AWAITS</h1>
      <p>Catastrophic Forgetting Detection for Free</p>

      <div className='hero-btns'>
        <Button
          onClick={event =>  window.location.href='/detector'}
          className='btns'
          buttonStyle='btn--outline'
          buttonSize='btn--large'
          path='/detector'
        >
          GET STARTED
        </Button>
        <Button
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'
          onClick={() => {openInNewTab("https://ultimatecourses.com/blog/programmatically-navigate-react-router");navigate('/')}}
        >
          WATCH DEMO <i className='far fa-play-circle' />
        </Button>
       
      </div>
    </div>
  );
}

export default HeroSection;