import React from "react";
import { createRoot } from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
} from "react-router-dom";
import Home from "./routes/Home";
import Help from "./routes/Help";
import Navbar from "./components/Navbar";
import Detector from "./routes/Detector";
import ErrorPage from "./routes/ErrorPage";
import "./App.css";
import Footer from "./components/Footer";
import Forget from "./routes/Forget";
import Experiments from "./routes/Experiments";
import CfdCifar10 from "./routes/CfdCifar10";
import CfdCifar100 from "./routes/CfdCifar100";
import CfdMNIST from "./routes/CfdMNIST";
import CfdSVHN from "./routes/CfdSVHN";
import CfoCifar10 from "./routes/CfoCifar10";
import CfoCifar100 from "./routes/CfoCifar100";
import CfoMNIST from "./routes/CfoMNIST";
import CfoSVHN from "./routes/CfoSVHN";




const AppLayout = () => {
  return(
    <>
    <Navbar/>
    <Outlet/>
    <Footer/>
    </>
  );
};


const router = createBrowserRouter([

  {
    element: <AppLayout/>,
    errorElement: <ErrorPage/>,
    children: [
      {
        path: "/",
        element: <Home/>,
      },
      {
        path: "detector",
        element: <Detector/>,
      },
      {
        path: "help",
        element: <Help/>,
      },
      {
        path: "forget",
        element: <Forget/>,
      },
      
      {
        path: "experiments",
        element: <Experiments/>,
      },
      {
        path: "CfdCifar10",
        element: <CfdCifar10/>,
      },
      {
        path: "CfdCifar100",
        element: <CfdCifar100/>,
      },
      {
        path: "CfdMNIST",
        element: <CfdMNIST/>,
      },
      {
        path: "CfdSVHN",
        element: <CfdSVHN/>,
      },
      {
        path: "CfoCifar10",
        element: <CfoCifar10/>,
      },
      {
        path: "CfoCifar100",
        element: <CfoCifar100/>,
      },
      {
        path: "CfoMNIST",
        element: <CfoMNIST/>,
      },
      {
        path: "CfoSVHN",
        element: <CfoSVHN/>,
      },
     
    ],

  },
  
]);

createRoot(document.getElementById("root")).render(
  <RouterProvider router={router} />
);