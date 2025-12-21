import { useState, useEffect, useCallback } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { 
  Paper, 
  CardActionArea, 
  CardMedia, 
  Grid, 
  TableContainer, 
  Table, 
  TableBody, 
  TableHead, 
  TableRow, 
  TableCell, 
  Button, 
  CircularProgress,
  Box,
  Chip
} from "@material-ui/core";
import { useDropzone } from 'react-dropzone';
import Clear from '@material-ui/icons/Clear';
import CloudUpload from '@material-ui/icons/CloudUpload';
import CheckCircle from '@material-ui/icons/CheckCircle';
import Spa from '@material-ui/icons/Spa';
import axios from "axios";

// Customized primary button
const ColorButton = withStyles((theme) => ({
  root: {
    color: '#ffffff',
    backgroundColor: '#4f46e5',
    borderRadius: '12px',
    padding: '12px 28px',
    fontSize: '15px',
    fontWeight: 600,
    textTransform: 'none',
    boxShadow: '0 4px 14px 0 rgba(79, 70, 229, 0.39)',
    transition: 'all 0.3s ease',
    '&:hover': {
      backgroundColor: '#4338ca',
      boxShadow: '0 6px 20px rgba(79, 70, 229, 0.5)',
      transform: 'translateY(-2px)',
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  grow: { flexGrow: 1 },

  // AppBar
  appbar: {
    background: 'linear-gradient(135deg, #1e293b 0%, #312e81 100%)',
    boxShadow: '0 10px 30px rgba(0,0,0,0.25)',
    color: 'white',
  },
  title: {
    fontWeight: 800,
    fontSize: '22px',
    letterSpacing: '0.6px',
    display: 'flex',
    alignItems: 'center',
  },
  titleIcon: {
    fontSize: '30px',
    marginRight: '10px',
    color: '#a5b4fc',
  },

  // Main container
  mainContainer: {
    background: 'radial-gradient(circle at top, #1e293b, #020617)',
    minHeight: "calc(100vh - 64px)",
    paddingTop: '50px',
    paddingBottom: '50px',
  },

  // Grid
  gridContainer: {
    justifyContent: "center",
    padding: "3em 1em",
  },
  buttonGrid: {
    maxWidth: "500px",
    width: "100%",
  },

  // Image Card
  imageCard: {
    margin: "auto",
    maxWidth: 520,
    backgroundColor: '#ffffff',
    borderRadius: '24px',
    overflow: 'hidden',
    boxShadow: '0 25px 60px rgba(0,0,0,0.25)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
    '&:hover': {
      transform: 'translateY(-6px)',
      boxShadow: '0 35px 80px rgba(0,0,0,0.3)',
    },
  },
  imageCardEmpty: { minHeight: '400px' },
  media: { height: 450, objectFit: 'cover' },

  // Dropzone
  dropzoneContainer: {
    minHeight: '360px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: '20px',
    border: '2px dashed #6366f1',
    padding: '50px',
    transition: 'all 0.25s ease',
    cursor: 'pointer',
    '&:hover': {
      backgroundColor: '#eef2ff',
      borderColor: '#4338ca',
    },
  },
  dropzoneActive: { backgroundColor: '#eef2ff', borderColor: '#4338ca' },
  uploadIcon: { fontSize: '72px', color: '#6366f1', marginBottom: '18px' },
  dropzoneText: { color: '#475569', fontWeight: 500, textAlign: 'center' },
  dropzoneSubtext: { color: '#94a3b8', marginTop: '8px', fontSize: '12px' },

  // Loader
  loader: { color: '#6366f1', marginBottom: '24px' },
  processingText: { color: '#475569', fontWeight: 600, letterSpacing: '0.4px' },

  // Results
  detail: {
    backgroundColor: '#ffffff',
    padding: '32px 24px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px',
  },
  resultHeader: { width: '100%', textAlign: 'center', marginBottom: '20px' },

  tableContainer: {
    backgroundColor: '#f8fafc',
    borderRadius: '16px',
    boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
  },
  table: { backgroundColor: 'transparent' },
  tableHead: { backgroundColor: '#e0e7ff' },
  tableRow: { '&:last-child td': { borderBottom: 0 } },
  tableCell: { fontSize: '18px', color: '#1e293b', fontWeight: 600, padding: '16px', borderBottom: '1px solid #e2e8f0' },
  tableCell1: { fontSize: '14px', color: '#475569', fontWeight: 600, padding: '12px 16px', textTransform: 'uppercase', letterSpacing: '0.5px' },

  // Confidence bar
  confidenceBar: { width: 'calc(100% - 32px)', height: '8px', backgroundColor: '#e2e8f0', borderRadius: '10px', overflow: 'hidden', margin: '16px' },
  confidenceFill: { height: '100%', background: 'linear-gradient(90deg, #4f46e5, #22c55e)', transition: 'width 0.7s ease-in-out', borderRadius: '10px' },

  // Chips
  successChip: { backgroundColor: '#dcfce7', color: '#166534', fontWeight: 600, padding: '6px 12px', fontSize: '14px' },

  // Clear button
  clearButton: {
    width: "100%",
    borderRadius: "12px",
    padding: "14px 32px",
    fontSize: "16px",
    fontWeight: 600,
    textTransform: 'none',
    backgroundColor: '#ef4444',
    color: '#ffffff',
    boxShadow: '0 4px 14px 0 rgba(239, 68, 68, 0.39)',
    transition: 'all 0.3s ease',
    '&:hover': {
      backgroundColor: '#dc2626',
      boxShadow: '0 6px 20px rgba(239, 68, 68, 0.5)',
      transform: 'translateY(-2px)',
    },
  },
}));

console.log("ENV API URL:", process.env.REACT_APP_API_URL);

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  let confidence = 0;

  const sendFile = useCallback(async () => {
    if (image && selectedFile) {
      let formData = new FormData();
      formData.append("file", selectedFile);
      try {
        let res = await axios({
          method: "post",
          url: process.env.REACT_APP_API_URL,
          data: formData,
        });
        if (res.status === 200) {
          setData(res.data);
        }
      } catch (error) {
        console.error("Error uploading file:", error);
      }
      setIsloading(false);
    }
  }, [image, selectedFile]);

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) return;
    setIsloading(true);
    sendFile();
  }, [preview, sendFile]);

  const onDrop = useCallback((acceptedFiles) => {
    if (!acceptedFiles || acceptedFiles.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(acceptedFiles[0]);
    setData(undefined);
    setImage(true);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp'] },
    multiple: false,
    maxSize: 5000000
  });

  if (data) confidence = (parseFloat(data.confidence) * 100).toFixed(2);

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography className={classes.title} variant="h6" noWrap>
            <Spa className={classes.titleIcon} />
            Crop Disease Prediction System
          </Typography>
          <div className={classes.grow} />
        </Toolbar>
      </AppBar>
      <Container maxWidth={false} className={classes.mainContainer} disableGutters={true}>
        <Grid className={classes.gridContainer} container direction="row" justifyContent="center" alignItems="center" spacing={3}>
          <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!image ? classes.imageCardEmpty : ''}`}>
              {image && (
                <CardActionArea>
                  <CardMedia className={classes.media} image={preview} component="img" title="Uploaded Plant Image" />
                </CardActionArea>
              )}
              {!image && (
                <CardContent>
                  <Box {...getRootProps()} className={`${classes.dropzoneContainer} ${isDragActive ? classes.dropzoneActive : ''}`}>
                    <input {...getInputProps()} />
                    <CloudUpload className={classes.uploadIcon} />
                    <Typography variant="body1" className={classes.dropzoneText}>
                      {isDragActive ? 'Drop the image here...' : 'Drag and drop a plant leaf image here or click to browse'}
                    </Typography>
                    <Typography variant="caption" className={classes.dropzoneSubtext}>
                      Supported formats: JPG, PNG, GIF (Max 5MB)
                    </Typography>
                  </Box>
                </CardContent>
              )}
              {data && (
                <CardContent className={classes.detail}>
                  <Box className={classes.resultHeader}>
                    <Chip icon={<CheckCircle />} label="Analysis Complete" className={classes.successChip} />
                  </Box>
                  <TableContainer component={Paper} className={classes.tableContainer}>
                    <Table className={classes.table} size="small" aria-label="results table">
                      <TableHead className={classes.tableHead}>
                        <TableRow className={classes.tableRow}>
                          <TableCell className={classes.tableCell1}>Detected Disease</TableCell>
                          <TableCell align="right" className={classes.tableCell1}>Confidence</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow className={classes.tableRow}>
                          <TableCell component="th" scope="row" className={classes.tableCell}>{data.class}</TableCell>
                          <TableCell align="right" className={classes.tableCell}>{confidence}%</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                    <Box className={classes.confidenceBar}>
                      <Box className={classes.confidenceFill} style={{ width: `${confidence}%` }} />
                    </Box>
                  </TableContainer>
                </CardContent>
              )}
              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress size={60} className={classes.loader} />
                  <Typography className={classes.processingText} variant="h6" noWrap>
                    Analyzing Image...
                  </Typography>
                </CardContent>
              )}
            </Card>
          </Grid>
          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton 
                variant="contained" 
                className={classes.clearButton} 
                color="primary" 
                component="span" 
                size="large" 
                onClick={clearData} 
                startIcon={<Clear fontSize="large" />}
              >
                Analyze Another Image
              </ColorButton>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );
};
