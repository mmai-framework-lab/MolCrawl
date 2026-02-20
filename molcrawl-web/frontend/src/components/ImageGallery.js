import React, { useState, useEffect } from 'react';
import { Card, Badge, Row, Col, Spinner, Alert, Modal, Button } from 'react-bootstrap';
import './ImageGallery.css';

/**
 * 画像ギャラリーコンポーネント
 * 指定されたモデルタイプの画像を一覧表示する
 */
const ImageGallery = ({ modelType, title = "Generated Images" }) => {
    const [images, setImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    const [showModal, setShowModal] = useState(false);

    useEffect(() => {
        if (!modelType) {
            setLoading(false);
            return;
        }

        fetchImages();
    }, [modelType]);

    const fetchImages = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/images/${modelType}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch images');
            }

            setImages(data.images || []);
        } catch (err) {
            console.error('Error fetching images:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleImageClick = (image) => {
        setSelectedImage(image);
        setShowModal(true);
    };

    const handleCloseModal = () => {
        setShowModal(false);
        setSelectedImage(null);
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatDate = (timestamp) => {
        return new Date(timestamp * 1000).toLocaleString();
    };

    if (loading) {
        return (
            <Card className="image-gallery-card">
                <Card.Header>
                    <h5><i className="fas fa-images me-2"></i>{title}</h5>
                </Card.Header>
                <Card.Body className="text-center py-4">
                    <Spinner animation="border" variant="primary" />
                    <p className="mt-2 mb-0">Loading images...</p>
                </Card.Body>
            </Card>
        );
    }

    if (error) {
        return (
            <Card className="image-gallery-card">
                <Card.Header>
                    <h5><i className="fas fa-images me-2"></i>{title}</h5>
                </Card.Header>
                <Card.Body>
                    <Alert variant="warning">
                        <Alert.Heading>No Images Available</Alert.Heading>
                        <p className="mb-0">
                            No visualization images found for this dataset. Images will appear here after:
                        </p>
                        <ul className="mt-2 mb-0">
                            <li>Running data preparation scripts</li>
                            <li>Completing model evaluation</li>
                            <li>Generating visualization reports</li>
                        </ul>
                    </Alert>
                </Card.Body>
            </Card>
        );
    }

    return (
        <>
            <Card className="image-gallery-card">
                <Card.Header className="d-flex justify-content-between align-items-center">
                    <h5><i className="fas fa-images me-2"></i>{title}</h5>
                    <Badge bg="secondary">{images.length} images</Badge>
                </Card.Header>
                <Card.Body>
                    {images.length === 0 ? (
                        <Alert variant="info">
                            <Alert.Heading>No Images Yet</Alert.Heading>
                            <p className="mb-0">
                                Visualization images will appear here after running data preparation or evaluation scripts.
                            </p>
                        </Alert>
                    ) : (
                        <Row className="g-3">
                            {images.map((image, index) => (
                                <Col key={index} xs={12} sm={6} md={4} lg={3}>
                                    <Card
                                        className="image-thumbnail-card h-100"
                                        onClick={() => handleImageClick(image)}
                                        style={{ cursor: 'pointer' }}
                                    >
                                        <div className="image-thumbnail-container">
                                            <img
                                                src={image.thumbnailPath}
                                                alt={image.filename}
                                                className="image-thumbnail"
                                                onError={(e) => {
                                                    e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Crect width="100" height="100" fill="%23f8f9fa"/%3E%3Ctext x="50" y="50" text-anchor="middle" dy=".3em" fill="%236c757d"%3ENo Preview%3C/text%3E%3C/svg%3E';
                                                }}
                                            />
                                        </div>
                                        <Card.Body className="p-2">
                                            <Card.Title className="h6 mb-1 text-truncate" title={image.filename}>
                                                {image.filename}
                                            </Card.Title>
                                            <Card.Text className="small text-muted mb-1">
                                                <div>{formatFileSize(image.size)}</div>
                                                <div>{formatDate(image.modified)}</div>
                                            </Card.Text>
                                        </Card.Body>
                                    </Card>
                                </Col>
                            ))}
                        </Row>
                    )}
                </Card.Body>
            </Card>

            {/* 画像詳細モーダル */}
            <Modal show={showModal} onHide={handleCloseModal} size="lg" centered>
                <Modal.Header closeButton>
                    <Modal.Title>{selectedImage?.filename}</Modal.Title>
                </Modal.Header>
                <Modal.Body className="text-center">
                    {selectedImage && (
                        <>
                            <img
                                src={selectedImage.webPath}
                                alt={selectedImage.filename}
                                className="img-fluid mb-3"
                                style={{ maxHeight: '70vh' }}
                            />
                            <Row className="text-start">
                                <Col sm={6}>
                                    <strong>File Size:</strong> {formatFileSize(selectedImage.size)}
                                </Col>
                                <Col sm={6}>
                                    <strong>Modified:</strong> {formatDate(selectedImage.modified)}
                                </Col>
                            </Row>
                        </>
                    )}
                </Modal.Body>
                <Modal.Footer>
                    <Button
                        variant="primary"
                        onClick={() => window.open(selectedImage?.webPath, '_blank')}
                    >
                        <i className="fas fa-external-link-alt me-2"></i>
                        Open in New Tab
                    </Button>
                    <Button variant="secondary" onClick={handleCloseModal}>
                        Close
                    </Button>
                </Modal.Footer>
            </Modal>
        </>
    );
};

export default ImageGallery;
